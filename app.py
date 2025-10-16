from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qa_chain import answer_query
import os
from dotenv import load_dotenv

load_dotenv()

API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", 8000))

app = FastAPI(title="Local RAG Knowledge-Base Search Engine")


class QueryRequest(BaseModel):
    query: str
    top_k: int = 4


@app.get("/")
def root():
    return {"status": "ok", "message": "Local RAG running"}


@app.post("/api/query")
def query_endpoint(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty")
    try:
        result = answer_query(req.query, top_k=req.top_k)
        return {"query": req.query, "answer": result["answer"], "sources": [{"page_content": d.page_content} for d in result["source_docs"]]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=API_HOST, port=API_PORT, reload=True)
