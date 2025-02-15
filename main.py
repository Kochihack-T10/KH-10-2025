from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import requests
from qdrant_client import QdrantClient, models
import google.generativeai as genai
from fastapi import File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import PyPDF2
import docx


# Qdrant Configuration
QDRANT_URL = "https://4faa90a2-d64a-482e-a2a6-158722ea8045.europe-west3-0.gcp.cloud.qdrant.io:6333"
QDRANT_COLLECTION = "documents"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Initialize FastAPI app
app = FastAPI(title="Brain of Organization")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini AI
genai.configure(api_key="AIzaSyBLTGlNz1RZpuvRfAd2Z1chAAeEJxcEOqk")
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# Models
class Query(BaseModel):
    text: str
    category: str

class DocumentRequest(BaseModel):
    title: str
    content: str
    category: str

# Helper Functions for Queries
def process_hr_query(text: str) -> str:
    """Processes HR-related queries using Gemini AI."""
    prompt = f"""
    You are an AI-powered HR assistant providing professional and accurate responses 
    to employee queries related to policies, payroll, leave, and company regulations.

    Query: {text}
    
    Answer:
    """
    return gemini_model.generate_content(prompt).text

def process_it_query(text: str) -> str:
    """Processes IT-related queries using Gemini AI."""
    prompt = f"""
    You are an AI-powered IT support assistant helping employees troubleshoot 
    technical issues, software, hardware, and security-related queries.

    Query: {text}
    
    Answer:
    """
    return gemini_model.generate_content(prompt).text

def process_finance_query(text: str) -> str:
    prompt = f"""You are an AI-powered Finance assistant helping with budget management, 
    expense reports, tax compliance, and financial policies.
    Query: {text}
    Answer:"""
    return gemini_model.generate_content(prompt).text

def process_legal_query(text: str) -> str:
    prompt = f"""You are an AI-powered Legal assistant providing guidance on compliance, 
    company policies, and legal risk assessments.
    Query: {text}
    Answer:"""
    return gemini_model.generate_content(prompt).text

def process_security_query(text: str) -> str:
    prompt = f"""You are an AI-powered Security & Data Protection assistant providing guidance 
    on cybersecurity policies, data privacy, and security best practices.
    Query: {text}
    Answer:"""
    return gemini_model.generate_content(prompt).text

def process_general_query(text: str) -> str:
    """Handles general queries using Gemini AI."""
    prompt = f"""
    You are an AI assistant providing general support and information 
    to employees based on their queries.

    Query: {text}
    
    Answer:
    """
    return gemini_model.generate_content(prompt).text


def get_embedding(text: str):
    """Generate embeddings using Google's Text Embedding 004 model."""
    response = requests.post(
        "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedText",
        headers={"Content-Type": "application/json"},
        params={"key": GEMINI_API_KEY},
        json={"text": text},
    )
    return response.json()["embedding"]

@app.post("/api/documents")
async def add_document(file: UploadFile = File(...), category: str = "general"):
    """Upload a document (PDF or DOCX), extract text, embed it, and store it in Qdrant."""
    
    
    if not file.filename.endswith((".pdf", ".docx")):
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are allowed")

  
    if file.filename.endswith(".pdf"):
        text = extract_text_from_pdf(file)
    elif file.filename.endswith(".docx"):
        text = extract_text_from_docx(file)

    
    embedding = get_embedding(text)
    
    payload = {
        "title": file.filename,
        "category": category,
        "content": text,
    }

    
    client.upsert(
        collection_name=QDRANT_COLLECTION,
        points=[
            models.PointStruct(id=file.filename, vector=embedding, payload=payload)
        ],
    )

    return {"message": "Document added successfully", "filename": file.filename}



def extract_text_from_pdf(file):
    """Extract text from an uploaded PDF file."""
    reader = PyPDF2.PdfReader(file.file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()



def extract_text_from_docx(file):
    """Extract text from an uploaded DOCX file."""
    doc = docx.Document(file.file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text.strip()

@app.get("/api/documents")
async def search_documents(query: str, category: Optional[str] = None):
    """Search for relevant documents using vector similarity and return them if explicitly requested."""
    
    query_embedding = get_embedding(query)

    filters = None
    if category:
        filters = models.Filter(
            must=[models.FieldCondition(key="category", match=models.MatchValue(value=category))]
        )

    search_result = client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=query_embedding,
        query_filter=filters,
        limit=3,
    )

    if not search_result:
        return {"message": "No relevant documents found"}

    doc_texts = "\n\n".join([hit.payload["content"] for hit in search_result])
    doc_titles = [hit.payload["title"] for hit in search_result]  

    prompt = f"Based on the following documents, answer this query: {query}\n\n{doc_texts}"
    response_text = gemini_model.generate_content(prompt).text

    
    if "send me the document" in query.lower() or "provide the document" in query.lower():
        return {
            "response": response_text,
            "documents": [{"title": title, "download_url": f"/api/download/{title}"} for title in doc_titles]
        }
    
    return {"response": response_text}

@app.get("/api/download/{filename}")
async def download_document(filename: str):
    """Retrieve the original uploaded document for the user."""
    file_path = os.path.join("uploaded_docs", filename)  # Modify based on your storage location
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="application/octet-stream", filename=filename)
    else:
        raise HTTPException(status_code=404, detail="File not found")



@app.post("/api/query")
async def handle_query(query: Query):
    category = query.category.lower()
    
    if category == "hr":
        response = process_hr_query(query.text)
    elif category == "it":
        response = process_it_query(query.text)
    elif category == "finance":
        response = process_finance_query(query.text)
    elif category == "legal":
        response = process_legal_query(query.text)
    elif category == "security":
        response = process_security_query(query.text)
    else:
        response = process_general_query(query.text)

    return {"response": response}

if _name_ == "_main_":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
