# 📄 RAG-powered Document QA App with Gemini & Streamlit

This project allows you to upload PDFs or Excel files and ask questions based on their content. It uses Google's Gemini model and LangChain's RAG (Retrieval-Augmented Generation) pipeline, powered by Streamlit.

---

## 🚀 Features

- 📁 Upload PDF or Excel documents
- 🤖 Ask questions and get contextual answers
- 🧠 Maintains memory across chats
- 💾 Saves past memory using FAISS vector store
- 🔍 Displays sources used to answer

---

## 🧰 Tech Stack

- [Streamlit](https://streamlit.io/)
- [LangChain](https://www.langchain.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Gemini Pro / Flash (via `langchain-google-genai`)](https://python.langchain.com/docs/integrations/llms/google_generative_ai/)
- Python 3.10+

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

# (Optional) Create virtual environment
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
"# Rag_Model" 
