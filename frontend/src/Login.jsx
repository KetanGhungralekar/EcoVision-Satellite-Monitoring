import React, { useState } from 'react';
import { Mail, Lock, ArrowRight, Satellite } from 'lucide-react';
import './Login.css';

export default function Login({ onLogin }) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = (e) => {
    e.preventDefault();
    setIsLoading(true);
    // Simulate network delay for effect
    setTimeout(() => {
      setIsLoading(false);
      onLogin();
    }, 1200);
  };

  return (
    <div className="login-container">
      <div className="login-bg-shapes">
        <div className="shape shape-1"></div>
        <div className="shape shape-2"></div>
        <div className="shape shape-3"></div>
      </div>
      
      <div className="login-card">
        <div className="login-header">
          <div className="login-icon-wrapper">
            <Satellite size={32} color="#ffffff" />
          </div>
          <h1 className="login-title">EcoVision AI</h1>
          <p className="login-subtitle">Spatial computing for planetary intelligence</p>
        </div>

        <form className="login-form" onSubmit={handleSubmit}>
          <div className="input-group">
            <label htmlFor="email">Email Address</label>
            <div className="input-wrapper">
              <Mail className="input-icon" size={20} />
              <input 
                type="email" 
                id="email"
                className="login-input" 
                placeholder="you@example.com" 
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
              />
            </div>
          </div>

          <div className="input-group">
            <label htmlFor="password">Password</label>
            <div className="input-wrapper">
              <Lock className="input-icon" size={20} />
              <input 
                type="password" 
                id="password"
                className="login-input" 
                placeholder="••••••••" 
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
            </div>
          </div>

          <div className="login-options">
            <label className="remember-me">
              <input type="checkbox" />
              <span>Remember me</span>
            </label>
            <a href="#" className="forgot-password">Forgot password?</a>
          </div>

          <button type="submit" className="login-btn" disabled={isLoading}>
            {isLoading ? 'Authenticating...' : 'Sign In'}
            {!isLoading && <ArrowRight size={18} />}
          </button>
        </form>

        <div className="login-footer">
          Don't have an account? <a href="#">Request access</a>
        </div>
      </div>
    </div>
  );
}
