import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import axios from 'axios';
import { useAuth } from '../contexts/AuthContext';
import toast from 'react-hot-toast';

const Challenges = () => {
  const { user } = useAuth();
  const [challenges, setChallenges] = useState([]);
  const [tasks, setTasks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedChallenge, setSelectedChallenge] = useState(null);

  useEffect(() => {
    fetchChallenges();
    fetchTasks();
  }, []);

  const fetchChallenges = async () => {
    try {
      const token = localStorage.getItem('token');
      const response = await axios.get('/api/challenges', {
        headers: { Authorization: `Bearer ${token}` }
      });
      setChallenges(response.data.data || []);
    } catch (error) {
      toast.error('Failed to load challenges');
    } finally {
      setLoading(false);
    }
  };

  const fetchTasks = async () => {
    try {
      const token = localStorage.getItem('token');
      const response = await axios.get('/api/tasks?status=pending', {
        headers: { Authorization: `Bearer ${token}` }
      });
      setTasks(response.data.data || []);
    } catch (error) {
      console.error('Failed to load tasks');
    }
  };

  const generateChallenge = async (taskId) => {
    try {
      const token = localStorage.getItem('token');
      toast.loading('Generating challenge...');
      const response = await axios.post('/api/challenges/generate', { taskId }, {
        headers: { Authorization: `Bearer ${token}` }
      });
      toast.dismiss();
      toast.success('Challenge generated! 🎮');
      fetchChallenges();
      setSelectedChallenge(response.data.data);
    } catch (error) {
      toast.dismiss();
      toast.error('Failed to generate challenge');
    }
  };

  const completeChallenge = async (challengeId, completionData) => {
    try {
      const token = localStorage.getItem('token');
      await axios.post(`/api/challenges/${challengeId}/complete`, completionData, {
        headers: { Authorization: `Bearer ${token}` }
      });
      toast.success('Challenge completed! 🏆');
      fetchChallenges();
      setSelectedChallenge(null);
    } catch (error) {
      toast.error('Failed to complete challenge');
    }
  };

  const getChallengeTypeIcon = (type) => {
    switch (type) {
      case 'quiz': return '❓';
      case 'puzzle': return '🧩';
      case 'coding_challenge': return '💻';
      case 'timed_completion': return '⏱️';
      case 'streak': return '🔥';
      default: return '🎯';
    }
  };

  const getDifficultyColor = (difficulty) => {
    switch (difficulty) {
      case 'easy': return 'from-cyan-400 to-cyan-500';
      case 'medium': return 'from-purple-400 to-purple-500';
      case 'hard': return 'from-pink-400 to-red-400';
      default: return 'from-purple-400 to-cyan-400';
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
          <h1 className="text-5xl font-bold mb-2 bg-gradient-to-r from-purple-400 to-cyan-400 bg-clip-text text-transparent">
            Challenges
          </h1>
          <p className="text-gray-300 text-lg">Transform your tasks into engaging challenges</p>
        </motion.div>

        {/* Active Challenges */}
        <div className="mb-8">
          <h2 className="text-2xl font-bold text-white mb-4">Active Challenges</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {challenges.filter(c => c.status === 'active' || c.status === 'pending').map((challenge, idx) => (
              <motion.div
                key={challenge._id}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: idx * 0.1 }}
                onClick={() => setSelectedChallenge(challenge)}
                className="card-modern cursor-pointer hover:scale-105 transition-all duration-300 glow-purple"
              >
                <div className="flex items-center gap-3 mb-4">
                  <div className="text-4xl">{getChallengeTypeIcon(challenge.type)}</div>
                  <div className="flex-1">
                    <h3 className="text-xl font-bold text-white">{challenge.title}</h3>
                    <p className="text-gray-400 text-sm">{challenge.taskId?.title || 'Task'}</p>
                  </div>
                </div>
                
                <div className="flex items-center gap-3 mb-4">
                  <span className={`px-3 py-1 rounded-lg bg-gradient-to-r ${getDifficultyColor(challenge.difficulty)} text-white text-xs font-semibold`}>
                    {challenge.difficulty}
                  </span>
                  <span className="text-gray-400 text-sm">
                    🎯 {challenge.pointsReward} points
                  </span>
                </div>

                {challenge.configuration?.timeLimit && (
                  <div className="text-gray-300 text-sm mb-4">
                    ⏱️ {challenge.configuration.timeLimit} minutes
                  </div>
                )}

                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={(e) => {
                    e.stopPropagation();
                    setSelectedChallenge(challenge);
                  }}
                  className="w-full btn-modern btn-primary glow-cyan"
                >
                  Start Challenge
                </motion.button>
              </motion.div>
            ))}
          </div>
        </div>

        {/* Generate from Tasks */}
        {tasks.length > 0 && (
          <div className="mb-8">
            <h2 className="text-2xl font-bold text-white mb-4">Generate Challenge from Task</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {tasks.slice(0, 4).map((task) => (
                <motion.div
                  key={task._id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  className="card-modern flex items-center justify-between"
                >
                  <div>
                    <h3 className="text-lg font-bold text-white">{task.title}</h3>
                    <p className="text-gray-400 text-sm">{task.description || 'No description'}</p>
                  </div>
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={() => generateChallenge(task._id)}
                    className="btn-modern btn-primary glow-purple"
                  >
                    🎮 Generate
                  </motion.button>
                </motion.div>
              ))}
            </div>
          </div>
        )}

        {/* Challenge Detail Modal */}
        {selectedChallenge && (
          <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4">
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              className="card-modern bg-gray-800 max-w-2xl w-full max-h-[90vh] overflow-y-auto"
            >
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-4">
                  <div className="text-5xl">{getChallengeTypeIcon(selectedChallenge.type)}</div>
                  <div>
                    <h2 className="text-3xl font-bold text-white">{selectedChallenge.title}</h2>
                    <p className="text-gray-400">{selectedChallenge.description}</p>
                  </div>
                </div>
                <button
                  onClick={() => setSelectedChallenge(null)}
                  className="text-gray-400 hover:text-white text-2xl"
                >
                  ✕
                </button>
              </div>

              <div className="grid grid-cols-2 gap-4 mb-6">
                <div className="p-4 bg-gradient-to-r from-purple-400/20 to-purple-500/20 rounded-lg">
                  <div className="text-gray-400 text-sm">Difficulty</div>
                  <div className={`text-lg font-bold bg-gradient-to-r ${getDifficultyColor(selectedChallenge.difficulty)} bg-clip-text text-transparent`}>
                    {selectedChallenge.difficulty}
                  </div>
                </div>
                <div className="p-4 bg-gradient-to-r from-cyan-400/20 to-cyan-500/20 rounded-lg">
                  <div className="text-gray-400 text-sm">Points Reward</div>
                  <div className="text-lg font-bold text-cyan-300">{selectedChallenge.pointsReward} pts</div>
                </div>
              </div>

              {selectedChallenge.configuration?.questions && (
                <div className="mb-6">
                  <h3 className="text-xl font-bold text-white mb-4">Quiz Questions</h3>
                  {selectedChallenge.configuration.questions.map((q, idx) => (
                    <div key={idx} className="mb-4 p-4 bg-gray-700 rounded-lg">
                      <p className="text-white font-semibold mb-2">{q.question}</p>
                      <div className="space-y-2">
                        {q.options?.map((opt, optIdx) => (
                          <div key={optIdx} className="p-2 bg-gray-600 rounded text-gray-300">
                            {opt}
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {selectedChallenge.status !== 'completed' && (
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => completeChallenge(selectedChallenge._id, {
                    score: 100,
                    accuracy: 100,
                    attemptsUsed: 1
                  })}
                  className="w-full btn-modern btn-primary glow-cyan py-4 text-lg"
                >
                  Complete Challenge
                </motion.button>
              )}

              {selectedChallenge.status === 'completed' && (
                <div className="p-4 bg-gradient-to-r from-cyan-400/20 to-purple-400/20 rounded-lg">
                  <div className="text-center">
                    <div className="text-4xl mb-2">🎉</div>
                    <div className="text-lg font-bold text-white">Challenge Completed!</div>
                    <div className="text-gray-300 mt-2">
                      Score: {selectedChallenge.completionData?.score || 0}%
                    </div>
                  </div>
                </div>
              )}
            </motion.div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Challenges;

