import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(errors="replace")


# ==========================================
# 1. LOAD ENVIRONMENT VARIABLES
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR.parent / ".env"

load_dotenv(ENV_PATH)

api_key = os.getenv("NVIDIA_API_KEY")

if not api_key:
    raise ValueError(f"NVIDIA_API_KEY not found in {ENV_PATH} or environment.")


# ==========================================
# 2. DOCUMENT LOADING
# ==========================================

pdf_path = BASE_DIR / "BTech_Mathematics_Question_Bank_110_Pages.pdf"

print(f"Loading document: {pdf_path}")

if not pdf_path.is_file():
    raise FileNotFoundError(f"PDF file not found: {pdf_path}")

loader = PyPDFLoader(str(pdf_path))
docs = loader.load()

if not docs or not any(doc.page_content.strip() for doc in docs):
    raise ValueError(f"No readable text found in PDF: {pdf_path}")

print(f"Loaded {len(docs)} document pages.")


# ==========================================
# 3. DOCUMENT CHUNKING
# ==========================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=80
)

chunks = text_splitter.split_documents(docs)

if not chunks:
    raise ValueError(f"No text chunks could be created from PDF: {pdf_path}")

print(f"Created {len(chunks)} text chunks.")


# ==========================================
# 4. EMBEDDINGS
# ==========================================

print("Creating embeddings...")

embeddings = FastEmbedEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

vector_size = len(embeddings.embed_query("Test embedding"))

print(f"Embedding dimension: {vector_size}")


# ==========================================
# 5. QDRANT VECTOR DATABASE
# ==========================================

print("Initializing Qdrant...")

client = QdrantClient(location=":memory:")

client.create_collection(
    collection_name="rag_collection",
    vectors_config=VectorParams(
        size=vector_size,
        distance=Distance.COSINE
    )
)

vectorstore = QdrantVectorStore(
    client=client,
    collection_name="rag_collection",
    embedding=embeddings
)

vectorstore.add_documents(chunks)

print("Document indexing completed!")


# ==========================================
# 6. NVIDIA LLM INITIALIZATION
# ==========================================

print("Initializing NVIDIA LLM...")

llm = ChatNVIDIA(
    model="nvidia/nemotron-3.5-lightning-30b-a3b",
    nvidia_api_key=api_key,
    temperature=0.2,
    max_completion_tokens=2500
)


# ==========================================
# 7. RETRIEVER
# ==========================================

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)


# ==========================================
# 8. RAG PROMPT
# ==========================================

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a patient tutor for B.Tech Mathematics.

        Answer using only the provided context from the B.Tech Mathematics
        question bank. Explain mathematical reasoning clearly, show relevant
        steps, and preserve the notation used in the source.

        If the answer is not available in the context, clearly say so rather
        than making up an answer. Cite the source page number when available.

        Context:
        {context}
        """
    ),
    ("human", "{question}")
])


# ==========================================
# 9. RAG CHAIN
# ==========================================

rag_chain = prompt | llm | StrOutputParser()


# ==========================================
# 10. QUERY FUNCTION
# ==========================================

def query_rag(question: str) -> str:
    question = question.strip()
    if not question:
        raise ValueError("Question must not be empty.")

    print("\n" + "=" * 60)
    print("Question:", question)
    print("=" * 60)

    # Retrieve documents
    retrieved_docs = retriever.invoke(question)
    if not retrieved_docs:
        response = "I couldn't find relevant information in the question bank."
        print("\nANSWER:")
        print(response)
        return response

    print("\nRetrieved Context:")
    context_parts = []

    for i, doc in enumerate(retrieved_docs, start=1):

        page_num = doc.metadata.get("page", "Unknown")
        if isinstance(page_num, int):
            page_num += 1
        context_parts.append(f"[Page {page_num}]\n{doc.page_content}")

        preview = (
            doc.page_content.strip()
            .replace("\n", " ")[:150]
        )

        print(f"\n[{i}] Page {page_num}: {preview}...")

    # Prepare context
    context = "\n\n".join(context_parts)

    # Generate answer
    print("\nGenerating answer using NVIDIA AI...")

    response = rag_chain.invoke({
        "context": context,
        "question": question
    })

    print("\nANSWER:")
    print(response)

    print("\n" + "=" * 60)

    return response


# ==========================================
# 11. MAIN PROGRAM
# ==========================================

def main() -> None:
    print("B.Tech Mathematics Question Bank assistant (type 'q' to quit)")

    while True:
        question = input("\nAsk a mathematics question: ").strip()
        if question.lower() in {"q", "quit", "exit"}:
            break
        if not question:
            continue

        query_rag(question)


if __name__ == "__main__":
    main()