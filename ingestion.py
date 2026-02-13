import os 
from typing import TypedDict 
from langgraph.graph import StateGraph, END 
from langchain_community.document_loaders import PyPDFLoader 
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_chroma import Chroma


# Define paths
PERSIST_DIRECTORY = "D:\\Project\\rag_embeddings"
PDF_FOLDER = "D:\\Project\\Project Data"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "legal_acts"

BATCH_SIZE = 50  # Number of chunks to add in one batch

# ------Graph state-------
class IngestState(TypedDict):
    docs: list
    chunks: list
    vector_store: object
    db_exists: bool

# -----Nodes-------

# Check existing database
def check_db(state: IngestState):
    if os.path.exists(PERSIST_DIRECTORY) and os.listdir(PERSIST_DIRECTORY):
        print("Chroma vector DB exists.")
        return {"db_exists": True}
    else:
        print("Chroma vector DB not found.")
        return {"db_exists": False}

# Load PDF files
def load_pdf(state: IngestState):
    all_docs = []
    print("Loading PDFs...")

    for file in os.listdir(PDF_FOLDER):
        if file.endswith(".pdf"):
            path = os.path.join(PDF_FOLDER, file)
            print(f"Loading: {file}")
            loader = PyPDFLoader(path)
            # Lazy load documents
            for doc in loader.lazy_load():
                all_docs.append(doc)

    return {"docs": all_docs}

# Split documents into chunks
def split_docs(state: IngestState):
    print("----Splitting the document----")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50
    )
    all_chunks = []
    for doc in state["docs"]:
        chunks = text_splitter.split_documents([doc])
        all_chunks.extend(chunks)
    if not all_chunks:
        raise Exception("Chunking failed")
    print(f"Chunking successful: {len(all_chunks)} chunks created")
    return {"chunks": all_chunks}

# Create Chroma vector store with batch insert
def create_vector(state: IngestState):
    print("Creating Chroma vector DB in batches...")
    embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embedding
    )

    # Process in batches
    chunks = state["chunks"]
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        vector_store.add_documents(batch)
        print(f"Batch {i // BATCH_SIZE + 1} persisted ({len(batch)} chunks)")

    print("All batches added to Chroma.")
    return {"vector_store": vector_store}

# Load existing Chroma vector store
def load_vector(state: IngestState):
    print("Loading existing Chroma vector DB...")
    embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embedding
    )
    return {"vector_store": vector_store}

# --------Graph-------
graph = StateGraph(IngestState)

graph.add_node("check_db", check_db)
graph.add_node("load_pdf", load_pdf)
graph.add_node("split_docs", split_docs)
graph.add_node("create_vector", create_vector)
graph.add_node("load_vector", load_vector)

graph.set_entry_point("check_db")

# Conditional branch
graph.add_conditional_edges(
    "check_db",
    lambda state: "load" if state["db_exists"] else "create",
    {
        "load": "load_vector",
        "create": "load_pdf"
    }
)

# Create path
graph.add_edge("load_pdf", "split_docs")
graph.add_edge("split_docs", "create_vector")
graph.add_edge("create_vector", END)

# Load path
graph.add_edge("load_vector", END)

ingestion_app = graph.compile()

# ---------- Run ----------
if __name__ == "__main__":
    print("\n--- LANGGRAPH CONDITIONAL INGESTION (Batch Mode) ---")
    result = ingestion_app.invoke({})
    print("\nDone.")
    print("✅ Ingestion complete")
    print("Vector DB:", result["vector_store"])
