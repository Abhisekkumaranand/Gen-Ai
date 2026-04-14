from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("ERROR: GOOGLE_API_KEY not found in .env file!")
    exit()


pdf_path = Path("./crimeReport.pdf")

if not pdf_path.exists():
    print(f"ERROR: PDF file not found at {pdf_path}")
    exit()

loader = PyPDFLoader(file_path=pdf_path)
docs = loader.load()


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

split_docs = text_splitter.split_documents(docs)

print("Number of original documents (pages):", len(docs))
print("Number of chunks after splitting:", len(split_docs))


print("Creating embeddings...")
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=api_key
)


print("Connecting to Qdrant...")
client = QdrantClient(url="http://localhost:6333")

print("Building vector database...")
vector_store = Qdrant.from_documents(
    documents=split_docs,
    embedding=embedding_model,
    url="http://localhost:6333",
    collection_name="crime_reports",
    force_recreate=True,
)

print("Document ingestion completed successfully!")


retriever = vector_store.as_retriever(search_kwargs={"k": 3})


query = input("\nAsk your question: ")


print("Searching for relevant information...")
retrieved_docs = retriever.invoke(query)


context = "\n\n".join([doc.page_content for doc in retrieved_docs])


llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    google_api_key=api_key,
    temperature=0.3,
    max_retries=2,
)


prompt = f"""You are a helpful AI assistant.

STRICT RULES:
1. Answer ONLY using the provided context.
2. Do NOT use your own knowledge.
3. If the answer is not in the context, say "I don't know".
4. Be concise and accurate.
5. Do not make up information.

Context:
{context}

Question:
{query}

Answer:"""


print("Generating answer...")
response = llm.invoke(prompt)


print("\n" + "="*50)
print("ANSWER:")
print("="*50)
print(response.content)
print("="*50 + "\n")