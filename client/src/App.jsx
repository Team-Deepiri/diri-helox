import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from 'react-query';
import { Toaster } from 'react-hot-toast';
import { AuthProvider } from './contexts/AuthContext';
import { SocketProvider } from './contexts/SocketContext';
import { AdventureProvider } from './contexts/AdventureContext';
import ProtectedRoute from './components/ProtectedRoute';
import Navbar from './components/Navbar';
import Footer from './components/Footer';

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
  return (
    <QueryClientProvider client={queryClient}>
      <AuthProvider>
        <SocketProvider>
          <AdventureProvider>
            <Router>
              <div className="min-h-screen bg-gray-50">
                <Navbar />
                <main className="pb-20">
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
                      background: '#363636',
                      color: '#fff',
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
  );
}

export default App;
