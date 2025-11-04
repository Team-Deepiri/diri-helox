import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import axios from 'axios';
import { useAuth } from '../contexts/AuthContext';
import toast from 'react-hot-toast';

const Leaderboard = () => {
  const { user } = useAuth();
  const [leaderboard, setLeaderboard] = useState([]);
  const [userRank, setUserRank] = useState(null);
  const [loading, setLoading] = useState(true);
  const [period, setPeriod] = useState('all');

  useEffect(() => {
    fetchLeaderboard();
    fetchUserRank();
  }, [period]);

  const fetchLeaderboard = async () => {
    try {
      const token = localStorage.getItem('token');
      const response = await axios.get(`/api/gamification/leaderboard?limit=100&period=${period}`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setLeaderboard(response.data.data || []);
    } catch (error) {
      toast.error('Failed to load leaderboard');
    } finally {
      setLoading(false);
    }
  };

  const fetchUserRank = async () => {
    try {
      const token = localStorage.getItem('token');
      const response = await axios.get('/api/gamification/rank', {
        headers: { Authorization: `Bearer ${token}` }
      });
      setUserRank(response.data.data.rank);
    } catch (error) {
      console.error('Failed to fetch user rank');
    }
  };

  const getRankIcon = (rank) => {
    if (rank === 1) return '🥇';
    if (rank === 2) return '🥈';
    if (rank === 3) return '🥉';
    return `#${rank}`;
  };

  const getRankColor = (rank) => {
    if (rank === 1) return 'from-yellow-400 to-yellow-500';
    if (rank === 2) return 'from-gray-300 to-gray-400';
    if (rank === 3) return 'from-orange-400 to-orange-500';
    if (rank <= 10) return 'from-purple-400 to-purple-500';
    return 'from-gray-600 to-gray-700';
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
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h1 className="text-5xl font-bold mb-2 bg-gradient-to-r from-purple-400 to-cyan-400 bg-clip-text text-transparent">
            Leaderboard
          </h1>
          <p className="text-gray-300 text-lg">Compete with the best performers</p>
        </motion.div>

        {/* Period Selector */}
        <div className="flex gap-4 mb-8">
          {['all', 'week', 'month'].map((p) => (
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

        {/* User Rank Card */}
        {userRank && (
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="card-modern bg-gradient-to-r from-purple-500/30 to-cyan-500/30 mb-8 p-6 glow-cyan"
          >
            <div className="flex items-center justify-between">
              <div>
                <div className="text-sm text-gray-300 mb-1">Your Rank</div>
                <div className="text-4xl font-bold text-white">#{userRank}</div>
              </div>
              <div className="text-right">
                <div className="text-sm text-gray-300 mb-1">Keep climbing!</div>
                <div className="text-2xl">🚀</div>
              </div>
            </div>
          </motion.div>
        )}

        {/* Leaderboard */}
        <div className="space-y-4">
          {leaderboard.map((entry, idx) => {
            const rank = idx + 1;
            const isCurrentUser = entry.userId?._id === user?.id;
            
            return (
              <motion.div
                key={entry._id || idx}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: idx * 0.05 }}
                className={`card-modern flex items-center gap-6 p-6 ${
                  isCurrentUser ? 'ring-2 ring-cyan-400 glow-cyan' : ''
                } ${rank <= 3 ? 'bg-gradient-to-r ' + getRankColor(rank) + '/20' : ''}`}
              >
                {/* Rank */}
                <div className={`text-3xl font-bold ${
                  rank <= 3 ? 'text-white' : 'text-gray-400'
                }`}>
                  {getRankIcon(rank)}
                </div>

                {/* Avatar & Name */}
                <div className="flex items-center gap-4 flex-1">
                  <div className={`w-12 h-12 rounded-full flex items-center justify-center text-xl font-bold ${
                    rank === 1 ? 'bg-gradient-to-r from-yellow-400 to-yellow-500' :
                    rank === 2 ? 'bg-gradient-to-r from-gray-300 to-gray-400' :
                    rank === 3 ? 'bg-gradient-to-r from-orange-400 to-orange-500' :
                    'bg-gradient-to-r from-purple-400 to-cyan-400'
                  }`}>
                    {entry.userId?.name?.charAt(0).toUpperCase() || 'U'}
                  </div>
                  <div>
                    <div className="font-bold text-white text-lg">
                      {entry.userId?.name || 'Anonymous'}
                      {isCurrentUser && <span className="ml-2 text-cyan-400">(You)</span>}
                    </div>
                    <div className="text-sm text-gray-400">
                      Level {entry.level} • {entry.stats?.tasksCompleted || 0} tasks completed
                    </div>
                  </div>
                </div>

                {/* Stats */}
                <div className="flex items-center gap-8">
                  <div className="text-center">
                    <div className="text-2xl font-bold bg-gradient-to-r from-purple-300 to-cyan-300 bg-clip-text text-transparent">
                      {entry.points.toLocaleString()}
                    </div>
                    <div className="text-xs text-gray-400">Points</div>
                  </div>
                  <div className="text-center">
                    <div className="text-lg font-bold text-white">
                      🔥 {entry.streaks?.daily?.current || 0}
                    </div>
                    <div className="text-xs text-gray-400">Streak</div>
                  </div>
                  <div className="text-center">
                    <div className="text-lg font-bold text-white">
                      ⭐ {entry.badges?.length || 0}
                    </div>
                    <div className="text-xs text-gray-400">Badges</div>
                  </div>
                </div>
              </motion.div>
            );
          })}
        </div>

        {leaderboard.length === 0 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="card-modern text-center py-20"
          >
            <div className="text-6xl mb-4">🏆</div>
            <h3 className="text-2xl font-bold text-gray-300 mb-2">No rankings yet</h3>
            <p className="text-gray-400">Be the first to complete tasks and climb the leaderboard!</p>
          </motion.div>
        )}
      </div>
    </div>
  );
};

export default Leaderboard;
