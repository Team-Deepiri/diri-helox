import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import axios from 'axios';
import { useAuth } from '../contexts/AuthContext';
import toast from 'react-hot-toast';

const TaskManagement = () => {
  const { user } = useAuth();
  const [tasks, setTasks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newTask, setNewTask] = useState({
    title: '',
    description: '',
    type: 'manual',
    priority: 'medium',
    estimatedDuration: 30
  });

  useEffect(() => {
    fetchTasks();
  }, []);

  const fetchTasks = async () => {
    try {
      const token = localStorage.getItem('token');
      const response = await axios.get('/api/tasks', {
        headers: { Authorization: `Bearer ${token}` }
      });
      setTasks(response.data.data || []);
    } catch (error) {
      toast.error('Failed to load tasks');
    } finally {
      setLoading(false);
    }
  };

  const createTask = async (e) => {
    e.preventDefault();
    try {
      const token = localStorage.getItem('token');
      await axios.post('/api/tasks', newTask, {
        headers: { Authorization: `Bearer ${token}` }
      });
      toast.success('Task created!');
      setShowCreateModal(false);
      setNewTask({ title: '', description: '', type: 'manual', priority: 'medium', estimatedDuration: 30 });
      fetchTasks();
    } catch (error) {
      toast.error('Failed to create task');
    }
  };

  const completeTask = async (taskId) => {
    try {
      const token = localStorage.getItem('token');
      await axios.post(`/api/tasks/${taskId}/complete`, {
        actualDuration: newTask.estimatedDuration
      }, {
        headers: { Authorization: `Bearer ${token}` }
      });
      toast.success('Task completed! 🎉');
      fetchTasks();
    } catch (error) {
      toast.error('Failed to complete task');
    }
  };

  const deleteTask = async (taskId) => {
    try {
      const token = localStorage.getItem('token');
      await axios.delete(`/api/tasks/${taskId}`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      toast.success('Task deleted');
      fetchTasks();
    } catch (error) {
      toast.error('Failed to delete task');
    }
  };

  const getPriorityColor = (priority) => {
    switch (priority) {
      case 'urgent': return 'from-red-400 to-pink-400';
      case 'high': return 'from-purple-400 to-pink-400';
      case 'medium': return 'from-purple-300 to-cyan-300';
      case 'low': return 'from-cyan-300 to-blue-300';
      default: return 'from-purple-300 to-cyan-300';
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'completed': return 'bg-gradient-to-r from-cyan-400 to-cyan-500';
      case 'in_progress': return 'bg-gradient-to-r from-purple-400 to-purple-500';
      case 'pending': return 'bg-gray-600';
      default: return 'bg-gray-600';
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-16 w-16 border-t-4 border-b-4 border-purple-400 border-cyan-400"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <div className="flex items-center justify-between mb-4">
            <div>
              <h1 className="text-5xl font-bold mb-2 bg-gradient-to-r from-purple-400 to-cyan-400 bg-clip-text text-transparent">
                Task Management
              </h1>
              <p className="text-gray-300 text-lg">Organize and gamify your productivity</p>
            </div>
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => setShowCreateModal(true)}
              className="btn-modern btn-primary px-6 py-3 glow-purple"
            >
              <span className="text-xl mr-2">➕</span>
              New Task
            </motion.button>
          </div>
        </motion.div>

        {/* Task Stats */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8"
        >
          {[
            { label: 'Total Tasks', value: tasks.length, icon: '📋', color: 'from-purple-400 to-purple-500' },
            { label: 'Pending', value: tasks.filter(t => t.status === 'pending').length, icon: '⏳', color: 'from-cyan-400 to-cyan-500' },
            { label: 'In Progress', value: tasks.filter(t => t.status === 'in_progress').length, icon: '⚡', color: 'from-purple-400 to-cyan-400' },
            { label: 'Completed', value: tasks.filter(t => t.status === 'completed').length, icon: '✅', color: 'from-cyan-400 to-cyan-500' }
          ].map((stat, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: idx * 0.1 }}
              className={`card-modern bg-gradient-to-br ${stat.color} p-6 text-white`}
            >
              <div className="text-4xl mb-2">{stat.icon}</div>
              <div className="text-3xl font-bold mb-1">{stat.value}</div>
              <div className="text-sm opacity-90">{stat.label}</div>
            </motion.div>
          ))}
        </motion.div>

        {/* Tasks Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {tasks.map((task, idx) => (
            <motion.div
              key={task._id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: idx * 0.1 }}
              className="card-modern hover:scale-105 transition-all duration-300 glow-cyan"
            >
              <div className="flex items-start justify-between mb-4">
                <div className="flex-1">
                  <h3 className="text-xl font-bold text-white mb-2">{task.title}</h3>
                  {task.description && (
                    <p className="text-gray-300 text-sm mb-3">{task.description}</p>
                  )}
                </div>
                <span className={`px-3 py-1 rounded-full text-xs font-semibold ${getStatusColor(task.status)}`}>
                  {task.status}
                </span>
              </div>

              <div className="flex items-center gap-4 mb-4">
                <div className={`px-3 py-1 rounded-lg bg-gradient-to-r ${getPriorityColor(task.priority)} text-white text-xs font-semibold`}>
                  {task.priority}
                </div>
                <div className="text-gray-400 text-sm">
                  ⏱️ {task.estimatedDuration} min
                </div>
                <div className="text-gray-400 text-sm">
                  🏷️ {task.type}
                </div>
              </div>

              {task.completionData && (
                <div className="mb-4 p-3 bg-gradient-to-r from-cyan-400/20 to-purple-400/20 rounded-lg">
                  <div className="text-sm text-gray-300">
                    Efficiency: <span className="font-bold text-cyan-300">{Math.round(task.completionData.efficiency)}%</span>
                  </div>
                </div>
              )}

              <div className="flex gap-2">
                {task.status !== 'completed' && (
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={() => completeTask(task._id)}
                    className="flex-1 btn-modern btn-primary px-4 py-2 text-sm glow-cyan"
                  >
                    ✓ Complete
                  </motion.button>
                )}
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => deleteTask(task._id)}
                  className="btn-modern btn-glass px-4 py-2 text-sm"
                >
                  🗑️
                </motion.button>
              </div>
            </motion.div>
          ))}
        </div>

        {tasks.length === 0 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-center py-20"
          >
            <div className="text-6xl mb-4">📝</div>
            <h3 className="text-2xl font-bold text-gray-300 mb-2">No tasks yet</h3>
            <p className="text-gray-400 mb-6">Create your first task to get started!</p>
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => setShowCreateModal(true)}
              className="btn-modern btn-primary glow-purple"
            >
              Create Task
            </motion.button>
          </motion.div>
        )}
      </div>

      {/* Create Task Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="card-modern bg-gray-800 max-w-md w-full"
          >
            <h2 className="text-2xl font-bold mb-4 bg-gradient-to-r from-purple-400 to-cyan-400 bg-clip-text text-transparent">
              Create New Task
            </h2>
            <form onSubmit={createTask}>
              <div className="mb-4">
                <label className="block text-gray-300 mb-2">Title</label>
                <input
                  type="text"
                  required
                  value={newTask.title}
                  onChange={(e) => setNewTask({ ...newTask, title: e.target.value })}
                  className="w-full px-4 py-2 rounded-lg bg-gray-700 border border-purple-400/30 text-white focus:border-purple-400 focus:outline-none"
                />
              </div>
              <div className="mb-4">
                <label className="block text-gray-300 mb-2">Description</label>
                <textarea
                  value={newTask.description}
                  onChange={(e) => setNewTask({ ...newTask, description: e.target.value })}
                  className="w-full px-4 py-2 rounded-lg bg-gray-700 border border-purple-400/30 text-white focus:border-purple-400 focus:outline-none"
                  rows="3"
                />
              </div>
              <div className="grid grid-cols-2 gap-4 mb-4">
                <div>
                  <label className="block text-gray-300 mb-2">Priority</label>
                  <select
                    value={newTask.priority}
                    onChange={(e) => setNewTask({ ...newTask, priority: e.target.value })}
                    className="w-full px-4 py-2 rounded-lg bg-gray-700 border border-purple-400/30 text-white focus:border-purple-400 focus:outline-none"
                  >
                    <option value="low">Low</option>
                    <option value="medium">Medium</option>
                    <option value="high">High</option>
                    <option value="urgent">Urgent</option>
                  </select>
                </div>
                <div>
                  <label className="block text-gray-300 mb-2">Duration (min)</label>
                  <input
                    type="number"
                    value={newTask.estimatedDuration}
                    onChange={(e) => setNewTask({ ...newTask, estimatedDuration: parseInt(e.target.value) })}
                    className="w-full px-4 py-2 rounded-lg bg-gray-700 border border-purple-400/30 text-white focus:border-purple-400 focus:outline-none"
                  />
                </div>
              </div>
              <div className="flex gap-3">
                <motion.button
                  type="submit"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className="flex-1 btn-modern btn-primary glow-purple"
                >
                  Create Task
                </motion.button>
                <motion.button
                  type="button"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => setShowCreateModal(false)}
                  className="btn-modern btn-glass"
                >
                  Cancel
                </motion.button>
              </div>
            </form>
          </motion.div>
        </div>
      )}
    </div>
  );
};

export default TaskManagement;

