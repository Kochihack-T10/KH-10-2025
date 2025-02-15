from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi import Depends
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

users_db = {
    "admin": {"username": "admin", "password": "admin123", "role": "admin"},
    "super_admin": {"username": "super_admin", "password": "superadmin123", "role": "super_admin"},
    "employee": {"username": "employee", "password": "user123", "role": "user"},
}


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/login")


# Initialize Gemini AI

gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# Models
class Query(BaseModel):
    text: str
    category: str

class User(BaseModel):
    username: str
    password: str
    role: str

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

@app.post("/api/register")
async def register(user: User):
    """Register a new user by storing username, password, and role in Qdrant."""
    # Ensure that the username is unique
    search_result = client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=[],
        query_filter=models.Filter(
            must=[models.FieldCondition(key="username", match=models.MatchValue(value=user.username))]
        ),
        limit=1,
    )
    
    if search_result:
        raise HTTPException(status_code=400, detail="Username already exists")
    
    # Create the user document
    payload = {"username": user.username, "password": user.password, "role": user.role}
    client.upsert(
        collection_name=QDRANT_COLLECTION,
        points=[models.PointStruct(id=user.username, vector=[], payload=payload)],
    )

    return {"message": "User registered successfully"}

@app.post("/api/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login and validate the credentials by checking the Qdrant database."""
    # Search for the user in the database
    search_result = client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=[],
        query_filter=models.Filter(
            must=[models.FieldCondition(key="username", match=models.MatchValue(value=form_data.username))]
        ),
        limit=1,
    )
    
    if not search_result or search_result[0].payload["password"] != form_data.password:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    # Store the user's role and username in session
    user_role = search_result[0].payload["role"]
    return {"message": "Login successful", "role": user_role}

def get_current_user(token: str = Depends(oauth2_scheme)):
    """Fetch current user data from Qdrant using the username stored in the token."""
    search_result = client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=[],
        query_filter=models.Filter(
            must=[models.FieldCondition(key="username", match=models.MatchValue(value=token))]
        ),
        limit=1,
    )
    
    if not search_result:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    return search_result[0].payload


@app.post("/api/documents")
async def add_document(file: UploadFile = File(...), category: str = "general", user=Depends(get_current_user)):
    """Allow admin and super admin roles to upload documents."""
    if user["role"] not in ["admin", "super_admin"]:
        raise HTTPException(status_code=403, detail="Access denied: Only admins can upload documents")

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
        points=[models.PointStruct(id=file.filename, vector=embedding, payload=payload)],
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



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
