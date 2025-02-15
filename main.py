from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi import Depends
from pydantic import BaseModel
from typing import Optional
from passlib.context import CryptContext
import os
import time
import requests
from qdrant_client import QdrantClient, models
import google.generativeai as genai
from fastapi import File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import PyPDF2
import docx
import uuid
import numpy as np


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
origins = [
    "http://localhost:5173",  # Frontend URL
    
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/login")

# Hugging Face API setup
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/intfloat/e5-large"
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

headers = {
    "Authorization": f"Bearer {HUGGINGFACE_API_KEY}"
}

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

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

class LoginRequest(BaseModel):
    username: str
    password: str



# Helper Functions for Queries
def process_hr_query(text: str) -> str:
   
    prompt = f"""
    You are an AI-powered HR assistant providing professional and accurate responses 
    to employee queries related to policies, payroll, leave, and company regulations.

    Query: {text}
    
    Answer:
    """
    return gemini_model.generate_content(prompt).text

def process_it_query(text: str) -> str:
    
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


def make_request_with_retry(data):
    retries = 5
    for i in range(retries):
        response = requests.post(HUGGINGFACE_API_URL, headers=headers, json=data)
        
        if response.status_code == 503:
            error_message = response.json()
            print(f"Error: {error_message['error']}. Retrying in {int(error_message['estimated_time'])} seconds...")
            time.sleep(error_message['estimated_time'])
        elif response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.status_code} - {response.text}")
            break

def get_embeddings_from_document(document_text, category="general"):
    sentences = [s.strip() for s in document_text.split('.') if s.strip()]  # Split document text into sentences

        # Select source sentence based on category
    category_source_sentences = {
        "hr": "This sentence represents an HR policy document.",
        "it": "This sentence describes IT and security-related documentation.",
        "finance": "This sentence explains financial statements and policies.",
        "legal": "This sentence refers to a legal compliance document.",
        "security": "This sentence is about cybersecurity policies and data protection.",
        "general": "This sentence represents a general corporate document."
    }

    source_sentence = category_source_sentences.get(category.lower(), "This sentence represents a general corporate document.")

    embeddings = []
    
    for sentence in sentences:
        if sentence.strip():
            data = {
                "inputs": {
                    "source_sentence": source_sentence,  
                    "sentences": [sentence.strip()]
                }
            }
            response_data = make_request_with_retry(data)
            if response_data:
                embeddings.append(response_data)  
    
    return embeddings

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

@app.post("/api/register")
async def register(user: User):
    # Check if username already exists using scroll
    all_users = client.scroll(
        collection_name=QDRANT_COLLECTION,
        limit=1000,  # Adjust limit as necessary
    )

    # Check if any user has the same username
    username_exists = any(
        u.payload["username"] == user.username for u in all_users if hasattr(u, 'payload')
    )

    if username_exists:
        raise HTTPException(status_code=400, detail="Username already exists")

    # Generate a random 512-dimensional vector for the user
    user_vector = np.random.rand(512).tolist()

    # Use UUID for a unique user ID
    user_id = str(uuid.uuid4())  

    # Hash the password before storing
    hashed_password = hash_password(user.password)

    payload = {"username": user.username, "password": hashed_password, "role": user.role}

    # Insert user data into Qdrant
    client.upsert(
        collection_name=QDRANT_COLLECTION,
        points=[models.PointStruct(id=user_id, vector=user_vector, payload=payload)],
    )

    return {"message": "User  registered successfully"}


@app.post("/api/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    # Fetch user details using `scroll()`, which supports filtering without a query vector
    scroll_result = client.scroll(
        collection_name=QDRANT_COLLECTION,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key="username", match=models.MatchValue(value=form_data.username))]
        ),
        limit=1,
    )

    # Check if a user was found
    if not scroll_result or len(scroll_result[0]) == 0:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    stored_user = scroll_result[0][0].payload  # Extract user payload
    stored_password = stored_user["password"]

    # Verify the password
    if not pwd_context.verify(form_data.password, stored_password):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    return {
        "message": "Login successful",
        "username": stored_user["username"],
        "role": stored_user["role"]
    }

def get_current_user(token: str = Depends(oauth2_scheme)):
 
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
    if user["role"] not in ["admin", "super_admin"]:
        raise HTTPException(status_code=403, detail="Access denied: Only admins can upload documents")

    if not file.filename.endswith((".pdf", ".docx")):
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are allowed")
    
    if file.filename.endswith(".pdf"):
        text = extract_text_from_pdf(file)
    elif file.filename.endswith(".docx"):
        text = extract_text_from_docx(file)
    
    # Convert text to embeddings using Hugging Face
    embedding = get_embeddings_from_document(text)
    
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
    reader = PyPDF2.PdfReader(file.file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()

def extract_text_from_docx(file):
    doc = docx.Document(file.file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text.strip()

@app.get("/api/documents")
async def search_documents(query: str, category: Optional[str] = None):
    query_embedding = get_embeddings_from_document(query)  # Get query embedding
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

    return {"documents": [{"title": title, "content": text} for title, text in zip(doc_titles, doc_texts)]}

@app.get("/api/download/{filename}")
async def download_document(filename: str):
    file_path = os.path.join("uploaded_docs", filename)  
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
