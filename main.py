from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi import Depends
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from typing import Optional
from passlib.context import CryptContext
import logging
import os
import time
import requests
from qdrant_client import QdrantClient, models
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
import google.generativeai as genai
from fastapi import File, UploadFile, HTTPException, Response, Request, Form
from fastapi.responses import FileResponse
import PyPDF2
import docx
import uuid
import numpy as np
import io



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


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/login")

sessions = {}

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


def search_qdrant_documents(query_embedding, category=None):
    global client
    
    # Ensure query_embedding is a list of 512 floats
    if isinstance(query_embedding, float):
        query_embedding = [query_embedding] * 512  # This is just for demonstration, it should be an actual 512-dimensional embedding
    
    # Check if the dimension matches (e.g., 512)
    if len(query_embedding) != 512:
        raise ValueError("The query embedding must be a 512-dimensional vector")
    
    # If a category is provided, create a filter
    filters = None
    if category:
        filters = models.Filter(
            must=[models.FieldCondition(key="category", match=models.MatchValue(value=category))]
        )
    
    # Perform the search with query embedding and filters (if any)
    results = client.search(
        collection_name="documents",  # Replace with your collection name
        query_vector=query_embedding,  # query_embedding should be a 512-dimensional list
        query_filter=filters,  # Apply filters if category is specified
        limit=5,  # Number of results to return
    )
    
    return results



def process_hr_query(text: str) -> str:
    
    query_embedding = get_embeddings_from_document(text, category="hr")  
    print("query_embedding: ", query_embedding)

   
    relevant_docs = search_qdrant_documents(query_embedding, "hr")  
    print("Relevant Documents: ", relevant_docs)


    search_result_text = "\n".join([result['payload']['text'] for result in relevant_docs])


    prompt = f"""
    You are an AI-powered HR assistant providing professional and accurate responses 
    to employee queries related to policies, payroll, leave, and company regulations.

    Answer the following query based on the previous search result and the user query.

    Previous Search Result: {search_result_text}

    Query: {text}

    Answer:
    """
    
    search_result_embedded = get_embeddings_from_document(search_result_text, category="hr")

    return gemini_model.generate_content(prompt, query_embedding, search_result_embedded).text


def process_it_query(text: str, category: str) -> str:

    query_embedding = get_embeddings_from_document(text, category="it")  
    print("query_embedding: ", query_embedding)

    relevant_docs = search_qdrant_documents(query_embedding, "it")  
    print("Relevant Documents: ", relevant_docs)

    search_result_text = "\n".join([result['payload']['text'] for result in relevant_docs])
    
    prompt = f"""
    You are an AI-powered IT support assistant helping employees troubleshoot 
    technical issues, software, hardware, and security-related queries.

    Answer the queries based on the previous search result and the user query.

    Previous Search Result: {search_result_text}

    Query: {text}

    Answer:
    """
    

    search_result_embedded = get_embeddings_from_document(search_result_text, category="it")

    return gemini_model.generate_content(prompt, query_embedding, search_result_embedded).text


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
                embeddings.append(response_data[0])  
    
    if embeddings:
        return np.mean(embeddings, axis=0).tolist()
    return np.zeros(512).tolist()


def hash_password(password: str) -> str:
    return pwd_context.hash(password)



@app.post("/api/register")
async def register(user: User):
    all_users = client.scroll(collection_name=QDRANT_COLLECTION, limit=1000)
    if any(u.payload["username"] == user.username for u in all_users if hasattr(u, 'payload')):
        raise HTTPException(status_code=400, detail="Username already exists")

    user_id = str(uuid.uuid4())  
    user_vector = [uuid.uuid4().int % 10 for _ in range(512)]
    hashed_password = pwd_context.hash(user.password)

    client.upsert(
        collection_name=QDRANT_COLLECTION,
        points=[models.PointStruct(id=user_id, vector=user_vector, payload={
            "username": user.username,
            "password": hashed_password,
            "role": user.role
        })],
    )

    return {"message": "User registered successfully"}

@app.post("/api/login")
async def login(response: Response, form_data: OAuth2PasswordRequestForm = Depends()):
    print(response)
    scroll_result = client.scroll(
        collection_name=QDRANT_COLLECTION,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key="username", match=models.MatchValue(value=form_data.username))]
        ),
        limit=1,
    )

    print(scroll_result)

    if not scroll_result or len(scroll_result[0]) == 0:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    stored_user = scroll_result[0][0].payload

    if not pwd_context.verify(form_data.password, stored_user["password"]):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    session_id = str(uuid.uuid4())  
    sessions[session_id] = {"username": stored_user["username"], "role": stored_user["role"]}
    response.set_cookie(key="session_id", value=session_id, httponly=True, secure=False, samesite="Lax")
    print("Cookie set:", response.headers.get("set-cookie"))

    return {"message": "Login successful", "username": stored_user["username"], "role": stored_user["role"]}

@app.get("/api/protected")
async def protected_route(request: Request):
    session_id = request.cookies.get("session_id")
    if not session_id or session_id not in sessions:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return {"message": "Access granted", "user": sessions[session_id]}

@app.get("/api/admin")
async def admin_route(request: Request):
    session_id = request.cookies.get("session_id")
    if not session_id or session_id not in sessions:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    user = sessions[session_id]
    if user["role"] not in ["admin", "superadmin"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return {"message": "Welcome, Admin", "user": user}

@app.get("/api/superadmin")
async def superadmin_route(request: Request):
    session_id = request.cookies.get("session_id")
    if not session_id or session_id not in sessions:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    user = sessions[session_id]
    if user["role"] != "superadmin":
        raise HTTPException(status_code=403, detail="Superadmin access required")

    return {"message": "Welcome, Superadmin", "user": user}

@app.post("/api/logout")
async def logout(response: Response, request: Request):
    session_id = request.cookies.get("session_id")
    if session_id and session_id in sessions:
        del sessions[session_id]
    response.delete_cookie("session_id")
    return {"message": "Logged out successfully"}


@app.post("/api/documents")
async def add_document(
    category: str = Form(...),  # Form data for category
    file: UploadFile = File(...),  # Form data for file
):
    print("Received request to upload document")
    
    if file is None:
        print("No file received in the request")
        raise HTTPException(status_code=400, detail="No file provided")

    # Check the file type (normalize to lowercase to handle .PDF or .DOCX extensions)
    if not file.filename.lower().endswith((".pdf", ".docx")):
        print("Invalid file type:", file.filename)
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are allowed")

    try:
        print("Received file:", file.filename)
        print("Category:", category)

        # Extract text based on file type
        text = extract_text_from_pdf(file) if file.filename.endswith(".pdf") else extract_text_from_docx(file)
        print("Extracted text length:", len(text))

        # Split the document into chunks of 512 characters each
        chunk_size = 512
        text_chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        print(f"Document split into {len(text_chunks)} chunks")

        embeddings = []
        for chunk in text_chunks:
            # Generate embedding for each chunk
            embedding = get_embeddings_from_document(chunk)
            print(f"Generated embedding for chunk (length {len(chunk)}): {embedding}")
            
            # Check if embedding is a float or int and handle accordingly
            if isinstance(embedding, (float, int)):
                print("Embedding is a number:", embedding)
                embedding = [embedding] * 512  # Expand scalar to a 512-dimensional vector (or handle accordingly)
            
            # Ensure the embedding is of the correct shape (e.g., 512-dimensional)
            if len(embedding) != 512:
                print(f"Embedding dimension mismatch: expected 512, got {len(embedding)}")
                raise ValueError(f"Expected 512-dimensional embedding, got {len(embedding)}")
            
            embeddings.append(embedding)

        # Store each chunk with its embedding
        for idx, chunk in enumerate(text_chunks):
            payload = {
                "title": file.filename,
                "category": category,
                "content": chunk,  # Store each chunk of content
            }


            point = PointStruct(
                id=str(uuid.uuid4()),  # Unique ID for each chunk
                vector=embeddings[idx],  # Embedding vector for the chunk
                payload=payload  # Payload containing content, title, and category
            )


            client.upsert(
                collection_name=QDRANT_COLLECTION,  # Replace with your collection name
                points=[point]
             )
        
        print(f"Stored chunk {idx + 1}/{len(text_chunks)}")

        return JSONResponse(content={"message": "Document added successfully", "filename": file.filename})

    except Exception as e:
        # Log the error instead of print
        logging.error(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing file")

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(io.BytesIO(file.file.read()))
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()

def extract_text_from_docx(file):
    doc = docx.Document(io.BytesIO(file.file.read()))
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
