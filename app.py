
import os
from typing import List
from pathlib import Path
from tempfile import NamedTemporaryFile

import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document


# Constant
MEMORY_PATH = "memory_store"

from dotenv import load_dotenv
import os

# 🔐 Load environment variables
load_dotenv()

# ✅ Fetch from env variable
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# 🤖 Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY
)

# 🔍 Initialize Gemini Embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)


# 🏗️ Data Models
class AnswerModel(BaseModel):
    answer: str = Field(..., description="Final answer to user's question")
    sources: List[str] = Field(..., description="Document chunks used to form the answer")


# 📄 Lazy PDF loader (page-by-page)
class LazyPDFLoader:
    def __init__(self, file_path):
        self.loader = PyPDFLoader(file_path)
        self.pages = self.loader.load_and_split()
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.pages):
            raise StopIteration
        page = self.pages[self.index]
        self.index += 1
        return page


# 📊 Lazy Excel loader (row-by-row)
class LazyExcelLoader:
    def __init__(self, file_path):
        self.loader = UnstructuredExcelLoader(file_path)
        self.rows = self.loader.load()
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.rows):
            raise StopIteration
        row = self.rows[self.index]
        self.index += 1
        return row


# 🧠 Unified document loader (returns lazy iterable)
def load_document(file_path: str):
    if file_path.endswith(".pdf"):
        return LazyPDFLoader(file_path)
    elif file_path.endswith(".xlsx") or file_path.endswith(".xls"):
        return LazyExcelLoader(file_path)
    else:
        raise ValueError("Unsupported file format")


# 🔁 Load → Split → Embed → FAISS Vectorstore
def embed_and_print_chunks(file_path: str, print_chunks: bool = False):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    all_chunks = []

    for doc in load_document(file_path):
        splits = splitter.split_documents([doc])
        if print_chunks:
            for chunk in splits:
                print(chunk.page_content)
                print("-" * 80)
        all_chunks.extend(splits)

    vectorstore = FAISS.from_documents(all_chunks, embeddings)
    return vectorstore


# 🧠 Memory Management Functions
@st.cache_resource
def initialize_memory_store():
    if os.path.exists(MEMORY_PATH):
        return FAISS.load_local(MEMORY_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        # Initialize an empty FAISS index
        memory_store = FAISS.from_texts([""], embeddings)
        # Remove the dummy empty text we just added
        memory_store.delete([memory_store.index_to_docstore_id[0]])
        return memory_store


def add_to_memory_store(question: str, answer: str):
    summary = answer[:300] + "..." if len(answer) > 300 else answer
    memory_doc = Document(page_content=f"Q: {question}\nA: {summary}")
    memory_store.add_documents([memory_doc])
    memory_store.save_local(MEMORY_PATH)


def retrieve_memory_context(query: str, k: int = 3):
    results = memory_store.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in results])


# ❓ Question Answering Function
def ask_question(vectorstore, query: str, k: int = 3) -> AnswerModel:
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(query)

    memory_context = retrieve_memory_context(query)

    doc_context = "\n\n".join([doc.page_content for doc in docs])
    full_prompt = (
        f"You are a helpful assistant. Use the documents and past memory to answer.\n\n"
        f"Memory:\n{memory_context}\n\n"
        f"Document Context:\n{doc_context}\n\n"
        f"Question: {query}\n"
    )

    result = llm.invoke(full_prompt)
    answer = result.content.strip()

    # Add to memory store
    add_to_memory_store(query, answer)

    return AnswerModel(
        answer=answer,
        sources=[doc.page_content[:100] + "..." for doc in docs]
    )


# 🎨 Streamlit UI
def main():
    # Initialize memory store
    global memory_store
    memory_store = initialize_memory_store()

    # Configure page
    st.set_page_config(page_title="Document QA with Gemini", page_icon="📄")

    # Sidebar
   

    # Main content
    st.title("📄 Document QA with Gemini")
    st.caption("Powered by Google Gemini - Ask questions about your documents")

    # File upload
    uploaded_file = st.file_uploader(
        "Choose a document (PDF or Excel)",
        type=["pdf", "xlsx", "xls"],
        help="Maximum file size: 200MB"
    )

    if uploaded_file is not None:
        # Display file info
        col1, col2 = st.columns([1, 3])
        with col1:
            st.success(f"📁 {uploaded_file.name}")
        with col2:
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
            st.caption(f"Size: {file_size:.2f} MB")

        # Load the document into vectorstore
        with st.spinner("Processing document... This may take a while for large files"):
            try:
                # Save uploaded file to a temporary location
                with NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                # Process the document
                vectorstore = embed_and_print_chunks(tmp_path)
                st.session_state.vectorstore = vectorstore
                st.success("Document processed successfully!")

            except Exception as e:
                st.error(f"Error processing document: {str(e)}")
                st.stop()
            finally:
                # Clean up the temporary file
                if 'tmp_path' in locals() and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        st.divider()

        # Chat interface
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                if message.get("sources"):
                    with st.expander("Sources used"):
                        for i, source in enumerate(message["sources"], 1):
                            st.markdown(f"**Source {i}:**")
                            st.write(source)
                            st.divider()

        # Question input
        if prompt := st.chat_input("Ask a question about the document"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)

            # Get response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = ask_question(vectorstore, prompt)

                        # Display assistant response
                        st.markdown(response.answer)

                        # Add assistant response to chat history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response.answer,
                            "sources": response.sources
                        })

                        # Display sources
                        with st.expander("Sources used"):
                            for i, source in enumerate(response.sources, 1):
                                st.markdown(f"**Source {i}:**")
                                st.write(source)
                                st.divider()

                    except Exception as e:
                        st.error(f"Error generating answer: {str(e)}")


if __name__ == "__main__":
    main()
