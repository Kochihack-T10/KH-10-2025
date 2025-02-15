export interface User {
  username: string;
  role: 'admin' | 'super_admin' | 'user';
}

export interface LoginResponse {
  message: string;
  role: string;
}

export interface Document {
  title: string;
  download_url?: string;
}

export interface QueryResponse {
  response: string;
  documents?: Document[];
}