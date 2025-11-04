import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import axios from 'axios';
import { useAuth } from '../contexts/AuthContext';
import toast from 'react-hot-toast';

const AnalyticsDashboard = () => {
  const { user } = useAuth();
  const [analytics, setAnalytics] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [period, setPeriod] = useState('week');

  useEffect(() => {
    fetchAnalytics();
    fetchStats();
  }, [period]);

  const fetchAnalytics = async () => {
    try {
      const token = localStorage.getItem('token');
      const days = period === 'week' ? 7 : period === 'month' ? 30 : 365;
      const response = await axios.get(`/api/analytics?days=${days}`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setAnalytics(response.data.data || []);
    } catch (error) {
      toast.error('Failed to load analytics');
    } finally {
      setLoading(false);
    }
  };

  const fetchStats = async () => {
    try {
      const token = localStorage.getItem('token');
      const response = await axios.get(`/api/analytics/stats?period=${period}`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setStats(response.data.data);
    } catch (error) {
      console.error('Failed to fetch stats');
    }
  };

  if (loading || !stats) {
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
          <h1 className="text-5xl font-bold mb-2 bg-gradient-to-r from-purple-400 to-cyan-400 bg-clip-text text-transparent">
            Analytics Dashboard
          </h1>
          <p className="text-gray-300 text-lg">Track your productivity and insights</p>
        </motion.div>

        {/* Period Selector */}
        <div className="flex gap-4 mb-8">
          {['day', 'week', 'month'].map((p) => (
            <motion.button
              key={p}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => setPeriod(p)}
              className={`px-6 py-3 rounded-lg font-semibold transition-all ${
                period === p
                  ? 'bg-gradient-to-r from-purple-400 to-cyan-400 text-white glow-purple'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              {p.charAt(0).toUpperCase() + p.slice(1)}
            </motion.button>
          ))}
        </div>

        {/* Key Stats */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          {[
            { label: 'Tasks Completed', value: stats.totalTasksCompleted, icon: '✅', color: 'from-cyan-400 to-cyan-500' },
            { label: 'Challenges Completed', value: stats.totalChallengesCompleted, icon: '🎮', color: 'from-purple-400 to-purple-500' },
            { label: 'Time Spent', value: `${Math.round(stats.totalTimeSpent / 60)}h`, icon: '⏱️', color: 'from-purple-400 to-cyan-400' },
            { label: 'Points Earned', value: stats.totalPointsEarned.toLocaleString(), icon: '⭐', color: 'from-cyan-400 to-cyan-500' }
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
        </div>

        {/* Efficiency Chart */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="card-modern mb-8 p-6"
        >
          <h2 className="text-2xl font-bold text-white mb-4">Average Efficiency</h2>
          <div className="flex items-end gap-2 h-64">
            {analytics.slice(-7).map((day, idx) => {
              const efficiency = day.metrics?.averageEfficiency || 0;
              return (
                <div key={idx} className="flex-1 flex flex-col items-center">
                  <motion.div
                    initial={{ height: 0 }}
                    animate={{ height: `${efficiency}%` }}
                    transition={{ delay: idx * 0.1, duration: 0.5 }}
                    className="w-full bg-gradient-to-t from-purple-400 to-cyan-400 rounded-t-lg mb-2"
                  />
                  <div className="text-xs text-gray-400">{efficiency}%</div>
                </div>
              );
            })}
          </div>
        </motion.div>

        {/* Task Types Distribution */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="card-modern p-6"
          >
            <h2 className="text-2xl font-bold text-white mb-4">Tasks by Type</h2>
            <div className="space-y-4">
              {Object.entries(stats.tasksByType || {}).map(([type, count], idx) => (
                <div key={idx}>
                  <div className="flex justify-between mb-2">
                    <span className="text-gray-300 capitalize">{type.replace('_', ' ')}</span>
                    <span className="text-white font-bold">{count}</span>
                  </div>
                  <div className="w-full bg-gray-700 rounded-full h-2">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${(count / stats.totalTasksCompleted) * 100}%` }}
                      transition={{ delay: idx * 0.1 }}
                      className="h-full bg-gradient-to-r from-purple-400 to-cyan-400 rounded-full"
                    />
                  </div>
                </div>
              ))}
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="card-modern p-6"
          >
            <h2 className="text-2xl font-bold text-white mb-4">Challenges by Type</h2>
            <div className="space-y-4">
              {Object.entries(stats.challengesByType || {}).map(([type, count], idx) => (
                <div key={idx}>
                  <div className="flex justify-between mb-2">
                    <span className="text-gray-300 capitalize">{type.replace('_', ' ')}</span>
                    <span className="text-white font-bold">{count}</span>
                  </div>
                  <div className="w-full bg-gray-700 rounded-full h-2">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${(count / stats.totalChallengesCompleted) * 100}%` }}
                      transition={{ delay: idx * 0.1 }}
                      className="h-full bg-gradient-to-r from-cyan-400 to-purple-400 rounded-full"
                    />
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
        </div>

        {/* Insights */}
        {analytics.length > 0 && analytics[0]?.insights && analytics[0].insights.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="card-modern p-6"
          >
            <h2 className="text-2xl font-bold text-white mb-4">💡 AI Insights</h2>
            <div className="space-y-4">
              {analytics[0].insights.map((insight, idx) => (
                <motion.div
                  key={idx}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: idx * 0.1 }}
                  className={`p-4 rounded-lg ${
                    insight.priority === 'high' 
                      ? 'bg-gradient-to-r from-purple-500/30 to-cyan-500/30' 
                      : 'bg-gray-700'
                  }`}
                >
                  <div className="font-semibold text-white mb-1">{insight.message}</div>
                  <div className="text-sm text-gray-400 capitalize">{insight.type.replace('_', ' ')}</div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        )}
      </div>
    </div>
  );
};

export default AnalyticsDashboard;

