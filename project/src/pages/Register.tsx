import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Lock, User, UserCog } from 'lucide-react';
import toast from 'react-hot-toast';
import { register } from '../api';

export default function Register() {
  const [username, setUsername] = useState('admin');
  const [password, setPassword] = useState('admin@123');
  const [role, setRole] = useState('admin');
  const navigate = useNavigate();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      await register(username, password, role);
      toast.success('Registration successful');
      navigate('/login');
    } catch (error) {
      toast.error('Registration failed');
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-background p-4">
      <div className="w-full max-w-md space-y-8 bg-surface p-8 rounded-lg shadow-md animate-fade-in">
        <div className="text-center">
          <h2 className="text-3xl font-bold text-primary">Create Account</h2>
          <p className="mt-2 text-text-secondary">Register for a new account</p>
        </div>
        <form className="mt-8 space-y-6" onSubmit={handleSubmit}>
          <div className="space-y-4">
            <div className="relative">
              <User className="absolute left-3 top-3 h-5 w-5 text-text-secondary" />
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                className="w-full pl-10 pr-3 py-2 bg-background border border-text-secondary rounded-md focus:border-primary focus:ring-1 focus:ring-primary"
                placeholder="Username"
                required
              />
            </div>
            <div className="relative">
              <Lock className="absolute left-3 top-3 h-5 w-5 text-text-secondary" />
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full pl-10 pr-3 py-2 bg-background border border-text-secondary rounded-md focus:border-primary focus:ring-1 focus:ring-primary"
                placeholder="Password"
                required
              />
            </div>
            <div className="relative">
              <UserCog className="absolute left-3 top-3 h-5 w-5 text-text-secondary" />
              <select
                value={role}
                onChange={(e) => setRole(e.target.value)}
                className="w-full pl-10 pr-3 py-2 bg-background border border-text-secondary rounded-md focus:border-primary focus:ring-1 focus:ring-primary"
                required
              >
                <option value="user">User</option>
                <option value="admin">Admin</option>
                <option value="super_admin">Super Admin</option>
              </select>
            </div>
          </div>
          <button
            type="submit"
            className="w-full py-3 px-4 bg-primary text-white font-semibold rounded-md hover:bg-opacity-90 transition-colors"
          >
            Register
          </button>
        </form>
      </div>
    </div>
  );
}