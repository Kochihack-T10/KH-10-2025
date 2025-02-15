import axios from 'axios';
import { LoginResponse, QueryResponse } from './types';

const API_URL = 'http://localhost:8000/api';

const api = axios.create({
  baseURL: API_URL,
});



export const login = async (username: string, password: string) => {
  const data = new URLSearchParams();
  data.append('username', username);
  data.append('password', password);

  const response = await axios.post('http://localhost:8000/api/login', data, {
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
  });

  return response.data;
};


export const register = async (username: string, password: string, role: string) => {
  const response = await api.post('/register', { username, password, role });
  return response.data;
};

export const submitQuery = async (text: string, category: string) => {
  const response = await api.post<QueryResponse>('/query', { text, category });
  return response.data;
};

export const uploadDocument = async (file: File, category: string) => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('category', category);
  const response = await api.post('/documents', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
};

export const searchDocuments = async (query: string, category?: string) => {
  const response = await api.get<QueryResponse>('/documents', {
    params: { query, category },
  });
  return response.data;
};