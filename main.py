from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import logging
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
import google.generativeai as genai

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")  
qdrant = QdrantClient(QDRANT_HOST, port=6333)

# Initialize Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI(title="Brain of Organization")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class Query(BaseModel):
    text: str

class Document(BaseModel):
    content: str
    category: str

@app.post("/api/store_document")
async def store_document(doc: Document):
    """Embeds the document and stores it in Qdrant."""
    embedding = generate_embedding(doc.content)
    
    qdrant.upsert(
        collection_name="documents",
        points=[
            PointStruct(
                id=hash(doc.content),
                vector=embedding,
                payload={"content": doc.content, "category": doc.category},
            )
        ]
    )

    return {"message": "Document stored successfully."}
@app.post("/api/query")
async def query_documents(query: Query):
    """Searches Qdrant for relevant documents and generates response using Gemini."""
    query_embedding = generate_embedding(query.text)
    
    results = qdrant.search(
        collection_name="documents",
        query_vector=query_embedding,
        limit=3
    )

    relevant_docs = [doc.payload["content"] for doc in results]
    response = generate_gemini_response(query.text, relevant_docs) 
    return {"response": response}
def generate_embedding(text: str):
    """Generates text embeddings using text-embedding-004."""
    from google.generativeai import embed_text
    return embed_text(model="text-embedding-004", text=text)["embedding"]

def generate_gemini_response(query: str, context: list):
    """Generates a response using Gemini AI given a query and relevant context."""
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = f"""
    You are an AI assistant. Use the following context to answer the user's query.
    Context:
    {context}
    User Query:
    {query}
    Answer:
    """
    response = model.generate_content(prompt)
    return response.text
