🚀 Retrieval-Augmented Generation (RAG) App with Gemini Pro on GKE

http://34.93.179.46/


This project implements a scalable Retrieval-Augmented Generation (RAG) system where users can upload documents and ask questions. The app retrieves relevant content using FAISS and answers using Gemini Pro LLM. It is deployed on Google Kubernetes Engine (GKE) with full CI/CD using GitHub Actions.

📌 Features
✅ Upload PDFs and ask context-based questions

✅ Document chunking and vector storage using FAISS

✅ Prompting handled by Gemini Pro via API

✅ Frontend built with Streamlit

✅ Backend powered by FastAPI

✅ Containerized with Docker

✅ Continuous Deployment via GitHub Actions

✅ Scalable deployment on Google Kubernetes Engine (GKE)

✅ Google Cloud Artifact Registry for container image storage

🛠️ Tech Stack
Layer	Technology
Frontend	Streamlit
Backend	FastAPI
Vector DB	FAISS
LLM API	Gemini Pro
Containerization	Docker
CI/CD	GitHub Actions
Cloud	Google Cloud Platform (GCP)
Orchestration	Kubernetes (GKE)
Image Registry	Google Artifact Registry

📂 Folder Structure
bash
Copy
Edit
📦 rag-app
 ┣ 📁 app
 ┃ ┣ 📜 app.py                 # Streamlit frontend
 ┃ ┣ 📜 backend.py             # FastAPI endpoints
 ┃ ┣ 📜 vector_store.py        # FAISS logic
 ┃ ┣ 📜 llm_interface.py       # Gemini Pro prompt handler
 ┃ ┗ 📜 utils.py               # Preprocessing utilities
 ┣ 📜 Dockerfile               # Docker config
 ┣ 📜 requirements.txt         # Python dependencies
 ┣ 📁 k8s                      # Kubernetes YAMLs
 ┃ ┣ 📜 namespace.yaml
 ┃ ┣ 📜 deployment.yaml
 ┃ ┗ 📜 service.yaml
 ┣ 📜 .github/workflows/deploy.yaml  # GitHub Actions CI/CD
 ┗ 📜 README.md
🚀 Setup & Deployment
1. Clone the Repo
bash
Copy
Edit
git clone https://github.com/yourusername/rag-app.git
cd rag-app
2. Add Your Environment Variables
Set your Gemini API key, etc.

3. Build Docker Image Locally (optional)
bash
Copy
Edit
docker build -t rag-app .
4. Kubernetes Deployment on GKE
bash
Copy
Edit
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
5. GitHub Actions CI/CD
Ensure the GitHub workflow pushes Docker image to Artifact Registry and deploys updated pods automatically.

🔗 Live Demo
Once deployed, access the app using the external IP of the GKE service:



📄 License
This project is under the MIT License.
