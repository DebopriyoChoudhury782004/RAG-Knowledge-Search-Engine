import os
import faiss
import pickle
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoConfig
)
import torch
from dotenv import load_dotenv

load_dotenv()

# Directories and models
FAISS_DIR = os.getenv("FAISS_INDEX_DIR", "faiss_index")
GEN_MODEL = os.getenv("LOCAL_GEN_MODEL", "google/flan-t5-small")
EMBEDDINGS_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

print(f"Loading generator model: {GEN_MODEL}...")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)

# Detect model type
config = AutoConfig.from_pretrained(GEN_MODEL)
if config.model_type in ["t5", "bart", "pegasus", "mbart", "flan-t5"]:
    model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL)
else:
    model = AutoModelForCausalLM.from_pretrained(GEN_MODEL)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Load FAISS index and documents
index = faiss.read_index(os.path.join(FAISS_DIR, "index.faiss"))
with open(os.path.join(FAISS_DIR, "docs.pkl"), "rb") as f:
    docs = pickle.load(f)

# Embeddings model
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)


def answer_query(query, top_k=4):
    """
    Retrieve relevant documents using FAISS, embed the query,
    and generate an answer using the specified model.
    """
    # Embed query and retrieve top_k docs
    q_vec = embeddings.embed_query(query)
    D, I = index.search(np.array([q_vec], dtype="float32"), top_k)
    retrieved_docs = [docs[i] for i in I[0]]
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # Build prompt
    prompt = f"Answer the question based on the context:\n\nContext: {context}\n\nQuestion: {query}\nAnswer:"

    # Tokenize input safely
    inputs = tokenizer(prompt, return_tensors="pt",
                       truncation=True, max_length=1024).to(device)

    # Generate answer
    if config.model_type in ["t5", "bart", "pegasus", "mbart", "flan-t5"]:
        outputs = model.generate(
            **inputs,
            max_new_tokens=256  # Generate 256 tokens beyond input
        )
    else:
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            pad_token_id=tokenizer.eos_token_id  # Required for causal models
        )

    # Decode generated tokens
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"answer": answer, "source_docs": retrieved_docs}
