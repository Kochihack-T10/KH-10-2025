import React, { useState } from 'react';
import { FileUp, Search, Send } from 'lucide-react';
import toast from 'react-hot-toast';
import { useAuthStore } from '../store';
import { submitQuery, uploadDocument, searchDocuments } from '../api';
import type { Document } from '../types';

const categories = ['HR', 'IT', 'Finance', 'Legal', 'Security', 'General'];

export default function Dashboard() {
  const user = useAuthStore((state) => state.user);
  const [query, setQuery] = useState('');
  const [category, setCategory] = useState('General');
  const [response, setResponse] = useState('');
  const [documents, setDocuments] = useState<Document[]>([]);
  const [file, setFile] = useState<File | null>(null);
  const [uploadCategory, setUploadCategory] = useState('General');

  const handleQuerySubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const result = await submitQuery(query, category);
      setResponse(result.response);
      if (result.documents) {
        setDocuments(result.documents);
      }
      toast.success('Query processed successfully');
    } catch (error) {
      toast.error('Failed to process query');
    }
  };

  const handleDocumentSearch = async () => {
    try {
      const result = await searchDocuments(query, category);
      setResponse(result.response);
      if (result.documents) {
        setDocuments(result.documents);
      }
      toast.success('Documents retrieved successfully');
    } catch (error) {
      toast.error('Failed to retrieve documents');
    }
  };

  const handleFileUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;

    try {
      await uploadDocument(file, uploadCategory);
      setFile(null);
      toast.success('Document uploaded successfully');
    } catch (error) {
      toast.error('Failed to upload document');
    }
  };

  return (
    <div className="min-h-screen bg-background p-4 md:p-8">
      <div className="max-w-6xl mx-auto space-y-8">
        <header className="bg-surface p-6 rounded-lg shadow-md">
          <h1 className="text-2xl font-bold text-primary">Welcome, {user?.username}</h1>
          <p className="text-text-secondary mt-2">Role: {user?.role}</p>
        </header>

        <section className="bg-surface p-6 rounded-lg shadow-md animate-fade-in">
          <h2 className="text-xl font-semibold text-primary mb-4">Submit Query</h2>
          <form onSubmit={handleQuerySubmit} className="space-y-4">
            <div className="flex flex-col md:flex-row gap-4">
              <select
                value={category}
                onChange={(e) => setCategory(e.target.value)}
                className="bg-background border border-text-secondary rounded-md p-2 focus:border-primary focus:ring-1 focus:ring-primary"
              >
                {categories.map((cat) => (
                  <option key={cat} value={cat}>
                    {cat}
                  </option>
                ))}
              </select>
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Enter your query..."
                className="flex-1 bg-background border border-text-secondary rounded-md p-2 focus:border-primary focus:ring-1 focus:ring-primary"
              />
              <button
                type="submit"
                className="bg-primary text-white px-6 py-2 rounded-md hover:bg-opacity-90 transition-colors flex items-center gap-2"
              >
                <Send className="h-4 w-4" />
                Submit
              </button>
              <button
                type="button"
                onClick={handleDocumentSearch}
                className="bg-surface border border-primary text-primary px-6 py-2 rounded-md hover:bg-primary hover:text-white transition-colors flex items-center gap-2"
              >
                <Search className="h-4 w-4" />
                Search Docs
              </button>
            </div>
          </form>
        </section>

        {(user?.role === 'admin' || user?.role === 'super_admin') && (
          <section className="bg-surface p-6 rounded-lg shadow-md animate-fade-in">
            <h2 className="text-xl font-semibold text-primary mb-4">Upload Document</h2>
            <form onSubmit={handleFileUpload} className="space-y-4">
              <div className="flex flex-col md:flex-row gap-4">
                <select
                  value={uploadCategory}
                  onChange={(e) => setUploadCategory(e.target.value)}
                  className="bg-background border border-text-secondary rounded-md p-2 focus:border-primary focus:ring-1 focus:ring-primary"
                >
                  {categories.map((cat) => (
                    <option key={cat} value={cat}>
                      {cat}
                    </option>
                  ))}
                </select>
                <input
                  type="file"
                  onChange={(e) => setFile(e.files?.[0] || null)}
                  accept=".pdf,.docx"
                  className="flex-1 bg-background border border-text-secondary rounded-md p-2 focus:border-primary focus:ring-1 focus:ring-primary"
                />
                <button
                  type="submit"
                  disabled={!file}
                  className="bg-primary text-white px-6 py-2 rounded-md hover:bg-opacity-90 transition-colors flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <FileUp className="h-4 w-4" />
                  Upload
                </button>
              </div>
            </form>
          </section>
        )}

        {response && (
          <section className="bg-surface p-6 rounded-lg shadow-md animate-slide-up">
            <h2 className="text-xl font-semibold text-primary mb-4">Response</h2>
            <div className="bg-background p-4 rounded-md">
              <p className="text-text-primary whitespace-pre-wrap">{response}</p>
            </div>
          </section>
        )}

        {documents.length > 0 && (
          <section className="bg-surface p-6 rounded-lg shadow-md animate-slide-up">
            <h2 className="text-xl font-semibold text-primary mb-4">Related Documents</h2>
            <div className="space-y-4">
              {documents.map((doc) => (
                <div
                  key={doc.title}
                  className="bg-background p-4 rounded-md flex justify-between items-center"
                >
                  <span className="text-text-primary">{doc.title}</span>
                  {doc.download_url && (
                    <a
                      href={doc.download_url}
                      className="text-primary hover:text-opacity-80 transition-colors"
                      download
                    >
                      Download
                    </a>
                  )}
                </div>
              ))}
            </div>
          </section>
        )}
      </div>
    </div>
  );
}