import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
import faiss
import pickle
from dotenv import load_dotenv

load_dotenv()

DOCS_DIR = "docs"
FAISS_DIR = os.getenv("FAISS_INDEX_DIR", "faiss_index")
# Experiment with different values
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
# Experiment with different values
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))


def load_documents(path):
    docs = []
    for file in os.listdir(path):
        full_path = os.path.join(path, file)
        if file.lower().endswith(".pdf"):
            loader = PyPDFLoader(full_path)
        elif file.lower().endswith(".txt"):
            loader = TextLoader(full_path)
        else:
            continue
        docs.extend(loader.load())
    return docs


def main():
    print("Loading documents...")
    raw_docs = load_documents(DOCS_DIR)

    print("Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    docs = splitter.split_documents(raw_docs)

    print("Computing embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name=os.getenv("EMBEDDING_MODEL"))
    vectors = [embeddings.embed_query(doc.page_content) for doc in docs]

    dim = len(vectors[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors).astype("float32"))

    os.makedirs(FAISS_DIR, exist_ok=True)
    faiss.write_index(index, os.path.join(FAISS_DIR, "index.faiss"))

    with open(os.path.join(FAISS_DIR, "docs.pkl"), "wb") as f:
        pickle.dump(docs, f)

    print(f"✅ Ingestion complete. FAISS index saved to {FAISS_DIR}")


if __name__ == "__main__":
    import numpy as np
    main()
