import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import axios from 'axios';
import { useAuth } from '../contexts/AuthContext';
import toast from 'react-hot-toast';

const GamificationDashboard = () => {
  const { user } = useAuth();
  const [profile, setProfile] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchProfile();
  }, []);

  const fetchProfile = async () => {
    try {
      const token = localStorage.getItem('token');
      const response = await axios.get('/api/gamification/profile', {
        headers: { Authorization: `Bearer ${token}` }
      });
      setProfile(response.data.data);
    } catch (error) {
      toast.error('Failed to load profile');
    } finally {
      setLoading(false);
    }
  };

  const checkBadges = async () => {
    try {
      const token = localStorage.getItem('token');
      const response = await axios.post('/api/gamification/badges/check', {}, {
        headers: { Authorization: `Bearer ${token}` }
      });
      if (response.data.data.awardedBadges.length > 0) {
        toast.success(`🎉 Earned ${response.data.data.awardedBadges.length} new badge(s)!`);
        fetchProfile();
      }
    } catch (error) {
      console.error('Failed to check badges');
    }
  };

  useEffect(() => {
    checkBadges();
  }, []);

  if (loading || !profile) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-16 w-16 border-t-4 border-b-4 border-purple-400 border-cyan-400"></div>
      </div>
    );
  }

  const xpPercentage = (profile.xp / profile.xpToNextLevel) * 100;

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
            Gamification Dashboard
          </h1>
          <p className="text-gray-300 text-lg">Track your progress and achievements</p>
        </motion.div>

        {/* Level & XP Card */}
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="card-modern bg-gradient-to-br from-purple-500/30 to-cyan-500/30 mb-8 p-8 glow-purple"
        >
          <div className="flex items-center justify-between mb-6">
            <div>
              <div className="text-6xl font-bold bg-gradient-to-r from-purple-300 to-cyan-300 bg-clip-text text-transparent mb-2">
                Level {profile.level}
              </div>
              <div className="text-gray-300">Rank #{profile.rank || 'N/A'}</div>
            </div>
            <div className="text-right">
              <div className="text-4xl font-bold text-white mb-2">{profile.points.toLocaleString()}</div>
              <div className="text-gray-300">Total Points</div>
            </div>
          </div>

          {/* XP Bar */}
          <div className="mb-2">
            <div className="flex justify-between text-sm text-gray-300 mb-2">
              <span>XP Progress</span>
              <span>{profile.xp} / {profile.xpToNextLevel} XP</span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-6 overflow-hidden">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${xpPercentage}%` }}
                transition={{ duration: 1, ease: "easeOut" }}
                className="h-full bg-gradient-to-r from-purple-400 via-cyan-400 to-purple-400 bg-[length:200%_100%] animate-gradient-x"
              />
            </div>
          </div>
        </motion.div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          {[
            { label: 'Tasks Completed', value: profile.stats.tasksCompleted, icon: '✅', color: 'from-cyan-400 to-cyan-500' },
            { label: 'Challenges Completed', value: profile.stats.challengesCompleted, icon: '🎮', color: 'from-purple-400 to-purple-500' },
            { label: 'Total Time Spent', value: `${Math.round(profile.stats.totalTimeSpent / 60)}h`, icon: '⏱️', color: 'from-purple-400 to-cyan-400' },
            { label: 'Average Efficiency', value: `${Math.round(profile.stats.averageEfficiency)}%`, icon: '📈', color: 'from-cyan-400 to-cyan-500' }
          ].map((stat, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: idx * 0.1 }}
              className={`card-modern bg-gradient-to-br ${stat.color} p-6 text-white`}
            >
              <div className="text-4xl mb-2">{stat.icon}</div>
              <div className="text-3xl font-bold mb-1">{stat.value}</div>
              <div className="text-sm opacity-90">{stat.label}</div>
            </motion.div>
          ))}
        </div>

        {/* Streaks */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="card-modern bg-gradient-to-br from-purple-500/20 to-purple-600/20 p-6"
          >
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xl font-bold text-white">🔥 Daily Streak</h3>
              <div className="text-3xl">🔥</div>
            </div>
            <div className="text-4xl font-bold text-purple-300 mb-2">
              {profile.streaks.daily.current} days
            </div>
            <div className="text-gray-400 text-sm">
              Longest: {profile.streaks.daily.longest} days
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="card-modern bg-gradient-to-br from-cyan-500/20 to-cyan-600/20 p-6"
          >
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xl font-bold text-white">📅 Weekly Streak</h3>
              <div className="text-3xl">📅</div>
            </div>
            <div className="text-4xl font-bold text-cyan-300 mb-2">
              {profile.streaks.weekly.current} weeks
            </div>
            <div className="text-gray-400 text-sm">
              Longest: {profile.streaks.weekly.longest} weeks
            </div>
          </motion.div>
        </div>

        {/* Badges */}
        <div className="mb-8">
          <h2 className="text-2xl font-bold text-white mb-4">Your Badges</h2>
          {profile.badges.length > 0 ? (
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
              {profile.badges.map((badge, idx) => (
                <motion.div
                  key={idx}
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: idx * 0.05 }}
                  className="card-modern text-center p-4 hover:scale-110 transition-transform glow-cyan"
                >
                  <div className="text-4xl mb-2">{badge.badgeIcon || '🏆'}</div>
                  <div className="text-sm font-semibold text-white">{badge.badgeName}</div>
                  <div className="text-xs text-gray-400 mt-1">
                    {new Date(badge.earnedAt).toLocaleDateString()}
                  </div>
                </motion.div>
              ))}
            </div>
          ) : (
            <div className="card-modern text-center py-12">
              <div className="text-6xl mb-4">🏆</div>
              <h3 className="text-xl font-bold text-gray-300 mb-2">No badges yet</h3>
              <p className="text-gray-400">Complete tasks and challenges to earn badges!</p>
            </div>
          )}
        </div>

        {/* Achievements */}
        {profile.achievements && profile.achievements.length > 0 && (
          <div>
            <h2 className="text-2xl font-bold text-white mb-4">Achievements</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {profile.achievements.map((achievement, idx) => (
                <motion.div
                  key={idx}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: idx * 0.1 }}
                  className="card-modern flex items-center gap-4 p-4"
                >
                  <div className="text-4xl">{achievement.achievementName.includes('Streak') ? '🔥' : '⭐'}</div>
                  <div className="flex-1">
                    <div className="font-bold text-white">{achievement.achievementName}</div>
                    <div className="w-full bg-gray-700 rounded-full h-2 mt-2">
                      <div
                        className="h-full bg-gradient-to-r from-purple-400 to-cyan-400 rounded-full"
                        style={{ width: `${achievement.progress}%` }}
                      />
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default GamificationDashboard;

