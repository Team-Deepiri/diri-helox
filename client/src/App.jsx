import React, { useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from 'react-query';
import { Toaster } from 'react-hot-toast';
import { AuthProvider } from './contexts/AuthContext';
import { SocketProvider } from './contexts/SocketContext';
import { AdventureProvider } from './contexts/AdventureContext';
import ProtectedRoute from './components/ProtectedRoute';
import ErrorBoundary from './components/ErrorBoundary';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import { setupGlobalErrorHandling, setupPerformanceMonitoring } from './utils/logger';

// Pages
import Home from './pages/Home';
import Login from './pages/Login';
import Register from './pages/Register';
import Dashboard from './pages/Dashboard';
import AdventureGenerator from './pages/AdventureGenerator';
import AdventureDetail from './pages/AdventureDetail';
import AdventureHistory from './pages/AdventureHistory';
import Events from './pages/Events';
import EventDetail from './pages/EventDetail';
import CreateEvent from './pages/CreateEvent';
import Profile from './pages/Profile';
import Friends from './pages/Friends';
import Leaderboard from './pages/Leaderboard';
import Notifications from './pages/Notifications';
import AgentChat from './pages/AgentChat';

// Create a client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

function App() {
  useEffect(() => {
    // Setup global error handling and performance monitoring
    setupGlobalErrorHandling();
    setupPerformanceMonitoring();
  }, []);

  return (
    <ErrorBoundary>
      <QueryClientProvider client={queryClient}>
        <AuthProvider>
          <SocketProvider>
            <AdventureProvider>
              <Router>
                <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-violet-900 relative overflow-hidden">
                  {/* Background Effects */}
                  <div className="fixed inset-0 bg-[url('data:image/svg+xml,%3Csvg width="60" height="60" viewBox="0 0 60 60" xmlns="http://www.w3.org/2000/svg"%3E%3Cg fill="none" fill-rule="evenodd"%3E%3Cg fill="%239C92AC" fill-opacity="0.05"%3E%3Ccircle cx="30" cy="30" r="2"/%3E%3C/g%3E%3C/g%3E%3C/svg%3E')] opacity-20" />
                  
                  {/* Gradient Orbs */}
                  <div className="fixed top-0 left-1/4 w-96 h-96 bg-purple-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-pulse" />
                  <div className="fixed top-0 right-1/4 w-96 h-96 bg-pink-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-pulse" style={{ animationDelay: '2s' }} />
                  <div className="fixed bottom-0 left-1/2 w-96 h-96 bg-blue-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-pulse" style={{ animationDelay: '4s' }} />
                  
                  <Navbar />
                  <main className="pt-20 pb-20 relative z-10">
                    <Routes>
                    {/* Public Routes */}
                    <Route path="/" element={<Home />} />
                    <Route path="/login" element={<Login />} />
                    <Route path="/register" element={<Register />} />
                    
                    {/* Protected Routes */}
                    <Route path="/dashboard" element={
                      <ProtectedRoute>
                        <Dashboard />
                      </ProtectedRoute>
                    } />
                    <Route path="/adventure/generate" element={
                      <ProtectedRoute>
                        <AdventureGenerator />
                      </ProtectedRoute>
                    } />
                    <Route path="/adventure/:id" element={
                      <ProtectedRoute>
                        <AdventureDetail />
                      </ProtectedRoute>
                    } />
                    <Route path="/adventures" element={
                      <ProtectedRoute>
                        <AdventureHistory />
                      </ProtectedRoute>
                    } />
                    <Route path="/events" element={
                      <ProtectedRoute>
                        <Events />
                      </ProtectedRoute>
                    } />
                    <Route path="/events/:id" element={
                      <ProtectedRoute>
                        <EventDetail />
                      </ProtectedRoute>
                    } />
                    <Route path="/events/create" element={
                      <ProtectedRoute>
                        <CreateEvent />
                      </ProtectedRoute>
                    } />
                    <Route path="/profile" element={
                      <ProtectedRoute>
                        <Profile />
                      </ProtectedRoute>
                    } />
                    <Route path="/friends" element={
                      <ProtectedRoute>
                        <Friends />
                      </ProtectedRoute>
                    } />
                    <Route path="/leaderboard" element={
                      <ProtectedRoute>
                        <Leaderboard />
                      </ProtectedRoute>
                    } />
                    <Route path="/notifications" element={
                      <ProtectedRoute>
                        <Notifications />
                      </ProtectedRoute>
                    } />
                    <Route path="/agent" element={
                      <ProtectedRoute>
                        <AgentChat />
                      </ProtectedRoute>
                    } />
                  </Routes>
                </main>
                <Footer />
                <Toaster
                  position="top-right"
                  toastOptions={{
                    duration: 4000,
                    style: {
                      background: 'rgba(0, 0, 0, 0.8)',
                      backdropFilter: 'blur(10px)',
                      color: '#fff',
                      border: '1px solid rgba(255, 255, 255, 0.2)',
                      borderRadius: '12px',
                    },
                    success: {
                      duration: 3000,
                      iconTheme: {
                        primary: '#10B981',
                        secondary: '#fff',
                      },
                    },
                    error: {
                      duration: 5000,
                      iconTheme: {
                        primary: '#EF4444',
                        secondary: '#fff',
                      },
                    },
                  }}
                />
              </div>
            </Router>
          </AdventureProvider>
        </SocketProvider>
      </AuthProvider>
    </QueryClientProvider>
    </ErrorBoundary>
  );
}

export default App;
