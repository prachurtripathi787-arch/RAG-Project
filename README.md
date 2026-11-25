# 📚 PDF-Based RAG Question Answering System

An AI-powered **Retrieval-Augmented Generation (RAG)** application built using Python, LangChain, NVIDIA AI, FastEmbed, and Qdrant.

This project allows users to ask questions from PDF documents and receive relevant, context-based answers using AI instead of manually searching through lengthy documents.

## 🚀 Project Overview

This project implements a Retrieval-Augmented Generation (RAG) pipeline that reads and processes PDF documents, converts their content into vector embeddings, stores them in a vector database, and retrieves relevant information to generate accurate answers.

The system uses a Statistical Techniques PDF as its knowledge source.

## ✨ Features

- 📄 PDF Document Loading using PyPDFLoader.
- ✂️ Text Chunking using RecursiveCharacterTextSplitter.
- 🧠 Semantic Embeddings using FastEmbed.
- 🔍 Similarity Search using Qdrant Vector Database.
- 🤖 AI-powered responses using NVIDIA Nemotron.
- 🔗 LangChain integration for RAG pipeline management.
- 🔐 Secure API key management using environment variables.
- ⚡ Context-aware question answering.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming Language |
| LangChain | RAG Pipeline |
| PyPDFLoader | PDF Document Loading |
| RecursiveCharacterTextSplitter | Text Chunking |
| FastEmbed | Text Embeddings |
| BAAI/bge-small-en-v1.5 | Embedding Model |
| Qdrant | Vector Database |
| NVIDIA AI | LLM Integration |
| Nemotron | Answer Generation |
| python-dotenv | Environment Variable Management |

## 🏗️ Project Architecture

The project follows these steps:

1. **Document Loading:** Load PDF documents using PyPDFLoader.
2. **Text Splitting:** Divide document content into smaller chunks.
3. **Embedding Generation:** Convert text chunks into numerical vectors using FastEmbed.
4. **Vector Storage:** Store embeddings in Qdrant.
5. **Query Processing:** Accept questions from the user.
6. **Similarity Search:** Retrieve the three most relevant document chunks.
7. **Answer Generation:** Send retrieved context to NVIDIA Nemotron.
8. **Final Response:** Generate a context-based answer.

## 📂 Project Structure

```text
RAG_Project/
│
├── main.py
├── .env
├── .gitignore
├── requirement.txt
│
├── document_loader/
│
└── STATS.pdf
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to Project Directory

```bash
cd RAG_Project
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirement.txt
```

### 5. Configure NVIDIA API Key

Create a `.env` file in the project directory.

Add your NVIDIA API key:

```env
NVIDIA_API_KEY=your_nvidia_api_key
```

Get your API key from:

https://build.nvidia.com/

### 6. Run the Project

```bash
python main.py
```

## 🧠 Models and Configuration

**Embedding Model:**
- Model: `BAAI/bge-small-en-v1.5`
- Vector Dimensions: 384
- Framework: FastEmbed

**Large Language Model:**
- Provider: NVIDIA
- Model: `nvidia/nemotron-3.5-lightning-30b-a3b`
- Temperature: 0.2
- Maximum Completion Tokens: 2500

**Vector Database:**
- Database: Qdrant
- Collection: `rag_collection`
- Distance Metric: Cosine Similarity
- Retrieval Count: 3

**Text Splitting:**
- Chunk Size: 600
- Chunk Overlap: 80

## 📦 Dependencies

The project uses the following Python libraries:

```text
python-dotenv
langchain
langchain-community
langchain-text-splitters
langchain-qdrant
qdrant-client
fastembed
langchain-nvidia-ai-endpoints
pypdf
```

Install all dependencies:

```bash
pip install -r requirement.txt
```

## 🔒 Security

- API keys are stored in the `.env` file.
- Sensitive credentials should never be uploaded to GitHub.
- The `.env` file should be included in `.gitignore`.

## 🎯 Applications

- Educational Question Answering
- PDF-Based AI Chatbots
- Academic Research Assistance
- Document Information Retrieval
- AI-Powered Study Assistants

## 🔮 Future Improvements

- Support for multiple PDF documents.
- Streamlit-based interactive user interface.
- Chat history and conversational memory.
- Persistent vector database storage.
- Improved document retrieval.
- Deployment using cloud platforms.

## 👨‍💻 Author

**Prachur Tripathi**

B.Tech Student | AI & Python Enthusiast

GitHub: [Your GitHub Profile](https://github.com/)

LinkedIn: [Prachur Tripathi](https://www.linkedin.com/in/prachur-tripathi-45209b322/)

---

⭐ If you find this project useful, consider giving it a star!