const Analytics = require('../models/Analytics');
const Task = require('../models/Task');
const Challenge = require('../models/Challenge');
const Gamification = require('../models/Gamification');
const logger = require('../utils/logger');

const analyticsService = {
  async recordTaskCompletion(userId, task) {
    try {
      const today = new Date();
      today.setHours(0, 0, 0, 0);

      let analytics = await Analytics.findOne({ userId, date: today });
      
      if (!analytics) {
        analytics = new Analytics({
          userId,
          date: today,
          metrics: {}
        });
      }

      analytics.metrics.tasksCompleted += 1;
      analytics.metrics.pointsEarned += 100; // Base points
      
      if (task.completionData?.actualDuration) {
        analytics.metrics.timeSpent += task.completionData.actualDuration;
      }

      // Update efficiency average
      const completedTasks = analytics.metrics.tasksCompleted;
      const currentAvg = analytics.metrics.averageEfficiency;
      const newEfficiency = task.completionData?.efficiency || 0;
      analytics.metrics.averageEfficiency = 
        ((currentAvg * (completedTasks - 1)) + newEfficiency) / completedTasks;

      // Track task type
      const taskType = task.type || 'manual';
      if (analytics.metrics.tasksByType[taskType] !== undefined) {
        analytics.metrics.tasksByType[taskType] += 1;
      }

      // Track peak productivity hour
      const hour = new Date().getHours();
      if (!analytics.metrics.peakProductivityHour) {
        analytics.metrics.peakProductivityHour = hour;
      }

      await analytics.save();
      await this.generateInsights(userId, analytics);
      
      return analytics;
    } catch (error) {
      logger.error('Error recording task completion:', error);
      throw error;
    }
  },

  async recordChallengeCompletion(userId, challenge) {
    try {
      const today = new Date();
      today.setHours(0, 0, 0, 0);

      let analytics = await Analytics.findOne({ userId, date: today });
      
      if (!analytics) {
        analytics = new Analytics({
          userId,
          date: today,
          metrics: {}
        });
      }

      analytics.metrics.challengesCompleted += 1;
      analytics.metrics.pointsEarned += challenge.pointsReward || 100;

      // Track challenge type
      const challengeType = challenge.type || 'timed_completion';
      if (analytics.metrics.challengesByType[challengeType] !== undefined) {
        analytics.metrics.challengesByType[challengeType] += 1;
      }

      await analytics.save();
      return analytics;
    } catch (error) {
      logger.error('Error recording challenge completion:', error);
      throw error;
    }
  },

  async getUserAnalytics(userId, days = 30) {
    try {
      const startDate = new Date();
      startDate.setDate(startDate.getDate() - days);
      startDate.setHours(0, 0, 0, 0);

      const analytics = await Analytics.find({
        userId,
        date: { $gte: startDate }
      }).sort({ date: -1 });

      return analytics;
    } catch (error) {
      logger.error('Error fetching user analytics:', error);
      throw error;
    }
  },

  async generateInsights(userId, analytics) {
    try {
      const insights = [];

      // Efficiency trend insight
      const recentAnalytics = await Analytics.find({
        userId,
        date: { $gte: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000) }
      }).sort({ date: -1 }).limit(7);

      if (recentAnalytics.length >= 3) {
        const efficiencies = recentAnalytics.map(a => a.metrics.averageEfficiency);
        const trend = efficiencies[0] > efficiencies[efficiencies.length - 1] ? 'improving' : 'declining';
        
        if (trend === 'improving') {
          insights.push({
            type: 'efficiency_trend',
            message: 'Your efficiency is improving! Keep up the great work!',
            data: { trend, efficiencies },
            priority: 'high'
          });
        }
      }

      // Peak hours insight
      const hourCounts = {};
      recentAnalytics.forEach(a => {
        const hour = a.metrics.peakProductivityHour;
        if (hour !== undefined) {
          hourCounts[hour] = (hourCounts[hour] || 0) + a.metrics.tasksCompleted;
        }
      });

      const peakHour = Object.keys(hourCounts).reduce((a, b) => 
        hourCounts[a] > hourCounts[b] ? a : b
      );

      if (peakHour) {
        insights.push({
          type: 'peak_hours',
          message: `You're most productive around ${peakHour}:00. Schedule important tasks then!`,
          data: { peakHour, hourCounts },
          priority: 'medium'
        });
      }

      // Task type preference
      const taskTypeTotals = {};
      recentAnalytics.forEach(a => {
        Object.keys(a.metrics.tasksByType || {}).forEach(type => {
          taskTypeTotals[type] = (taskTypeTotals[type] || 0) + a.metrics.tasksByType[type];
        });
      });

      const preferredType = Object.keys(taskTypeTotals).reduce((a, b) => 
        taskTypeTotals[a] > taskTypeTotals[b] ? a : b
      );

      if (preferredType && taskTypeTotals[preferredType] > 5) {
        insights.push({
          type: 'task_type_preference',
          message: `You complete more ${preferredType} tasks. Consider creating more of these!`,
          data: { preferredType, counts: taskTypeTotals },
          priority: 'low'
        });
      }

      // Update analytics with insights
      analytics.insights = insights;
      await analytics.save();

      return insights;
    } catch (error) {
      logger.error('Error generating insights:', error);
      return [];
    }
  },

  async getProductivityStats(userId, period = 'week') {
    try {
      const now = new Date();
      let startDate;

      switch (period) {
        case 'day':
          startDate = new Date(now.getFullYear(), now.getMonth(), now.getDate());
          break;
        case 'week':
          startDate = new Date(now);
          startDate.setDate(startDate.getDate() - 7);
          break;
        case 'month':
          startDate = new Date(now);
          startDate.setMonth(startDate.getMonth() - 1);
          break;
        default:
          startDate = new Date(now);
          startDate.setDate(startDate.getDate() - 7);
      }

      const analytics = await Analytics.find({
        userId,
        date: { $gte: startDate }
      });

      const stats = {
        totalTasksCompleted: analytics.reduce((sum, a) => sum + a.metrics.tasksCompleted, 0),
        totalChallengesCompleted: analytics.reduce((sum, a) => sum + a.metrics.challengesCompleted, 0),
        totalTimeSpent: analytics.reduce((sum, a) => sum + a.metrics.timeSpent, 0),
        totalPointsEarned: analytics.reduce((sum, a) => sum + a.metrics.pointsEarned, 0),
        averageEfficiency: analytics.length > 0
          ? analytics.reduce((sum, a) => sum + a.metrics.averageEfficiency, 0) / analytics.length
          : 0,
        tasksByType: {},
        challengesByType: {}
      };

      // Aggregate task types
      analytics.forEach(a => {
        Object.keys(a.metrics.tasksByType || {}).forEach(type => {
          stats.tasksByType[type] = (stats.tasksByType[type] || 0) + a.metrics.tasksByType[type];
        });
      });

      // Aggregate challenge types
      analytics.forEach(a => {
        Object.keys(a.metrics.challengesByType || {}).forEach(type => {
          stats.challengesByType[type] = (stats.challengesByType[type] || 0) + a.metrics.challengesByType[type];
        });
      });

      return stats;
    } catch (error) {
      logger.error('Error fetching productivity stats:', error);
      throw error;
    }
  }
};

module.exports = analyticsService;

