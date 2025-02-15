# Brain of Organization - AI Chatbot

This is a FastAPI-based internal AI chatbot designed for organizations to handle HR, IT, Finance, Legal, Security, and General queries. It integrates AI-powered responses, document management, and role-based access control.

---

## Features

- **User Authentication**: Secure login and registration with hashed passwords.
- **AI Query Handling**: Processes employee queries in different organizational domains.
- **Document Management**: Supports PDF and DOCX uploads, stores embeddings in Qdrant, and enables document search.
- **Access Control**: Role-based permissions for uploading and retrieving documents.
- **Vector Search**: Uses embeddings to retrieve relevant documents.
- **CORS Support**: Allows interaction with a frontend application.

---

## Qdrant Database

The system utilizes **Qdrant** as a vector database to store document embeddings and user data efficiently. Key aspects include:

- **High-Performance Vector Search**: Enables fast retrieval of relevant documents based on AI-generated embeddings.
- **Scalability**: Supports large-scale document storage and querying.
- **Role-Based Access Control**: Ensures secure data retrieval and storage.

---

## Gemini AI Model

The chatbot leverages **Gemini AI** to generate responses and process queries effectively. Features include:

- **Domain-Specific Query Handling**: AI-generated responses tailored to HR, IT, Finance, Legal, Security, and General queries.
- **Context Awareness**: Enhances the relevance of responses by understanding the nature of employee queries.
- **Integration with Document Search**: AI can cross-reference documents to provide enriched responses.

---

## Security Features

To ensure secure interactions and data protection, the system includes:

- **Hashed Passwords**: Uses bcrypt for password encryption.
- **Role-Based Authorization**: Restricts access to certain functionalities based on user roles.
- **Data Encryption**: Ensures stored user and document data are protected.
- **Secure API Endpoints**: Implements authentication mechanisms to prevent unauthorized access.

---

## API Endpoints

### Authentication

- **Register**: `POST /api/register`
- **Login**: `POST /api/login`

### Query Handling

- **Ask a question**: `POST /api/query`

### Document Management

- **Upload document**: `POST /api/documents`
- **Search documents**: `GET /api/documents`
- **Download document**: `GET /api/download/{filename}`

---

## Flow Diagram

![Flow Diagram](https://github.com/Kochihack-T10/KH-10-2025/blob/main/flow.jpg)

---

## Future Improvements

- Implement JWT authentication for better security.
- Enhance document retrieval with hybrid search.
- Expand support for additional document formats.
- Improve the UI for a better user experience.

---

## License

MIT License. Free to use and modify.

