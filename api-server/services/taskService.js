const Task = require('../models/Task');
const Challenge = require('../models/Challenge');
const Gamification = require('../models/Gamification');
const analyticsService = require('./analyticsService');
const logger = require('../utils/logger');

const taskService = {
  async createTask(userId, taskData) {
    try {
      const task = new Task({
        userId,
        ...taskData
      });
      await task.save();
      logger.info(`Task created: ${task._id} for user: ${userId}`);
      return task;
    } catch (error) {
      logger.error('Error creating task:', error);
      throw error;
    }
  },

  async getUserTasks(userId, filters = {}) {
    try {
      const query = { userId };
      
      if (filters.status) {
        query.status = filters.status;
      }
      if (filters.type) {
        query.type = filters.type;
      }
      if (filters.search) {
        query.$or = [
          { title: { $regex: filters.search, $options: 'i' } },
          { description: { $regex: filters.search, $options: 'i' } }
        ];
      }

      const tasks = await Task.find(query)
        .sort({ createdAt: -1 })
        .limit(filters.limit || 50)
        .skip(filters.skip || 0);
      
      return tasks;
    } catch (error) {
      logger.error('Error fetching tasks:', error);
      throw error;
    }
  },

  async getTaskById(taskId, userId) {
    try {
      const task = await Task.findOne({ _id: taskId, userId });
      if (!task) {
        throw new Error('Task not found');
      }
      return task;
    } catch (error) {
      logger.error('Error fetching task:', error);
      throw error;
    }
  },

  async updateTask(taskId, userId, updateData) {
    try {
      const task = await Task.findOneAndUpdate(
        { _id: taskId, userId },
        { $set: updateData },
        { new: true, runValidators: true }
      );
      
      if (!task) {
        throw new Error('Task not found');
      }
      
      logger.info(`Task updated: ${taskId} for user: ${userId}`);
      return task;
    } catch (error) {
      logger.error('Error updating task:', error);
      throw error;
    }
  },

  async deleteTask(taskId, userId) {
    try {
      const task = await Task.findOneAndDelete({ _id: taskId, userId });
      if (!task) {
        throw new Error('Task not found');
      }
      
      // Also delete associated challenge if exists
      if (task.challengeId) {
        await Challenge.findByIdAndDelete(task.challengeId);
      }
      
      logger.info(`Task deleted: ${taskId} for user: ${userId}`);
      return task;
    } catch (error) {
      logger.error('Error deleting task:', error);
      throw error;
    }
  },

  async completeTask(taskId, userId, completionData) {
    try {
      const task = await Task.findOne({ _id: taskId, userId });
      if (!task) {
        throw new Error('Task not found');
      }

      const actualDuration = completionData.actualDuration || 0;
      const estimatedDuration = task.estimatedDuration || 1;
      const efficiency = Math.min(100, Math.max(0, (estimatedDuration / actualDuration) * 100));

      task.status = 'completed';
      task.completionData = {
        completedAt: new Date(),
        actualDuration,
        efficiency,
        notes: completionData.notes || ''
      };

      await task.save();

      // Award points and update gamification
      await this.awardTaskCompletion(userId, task);

      // Record analytics
      await analyticsService.recordTaskCompletion(userId, task);

      logger.info(`Task completed: ${taskId} for user: ${userId}`);
      return task;
    } catch (error) {
      logger.error('Error completing task:', error);
      throw error;
    }
  },

  async awardTaskCompletion(userId, task) {
    try {
      let gamification = await Gamification.findOne({ userId });
      
      if (!gamification) {
        gamification = new Gamification({ userId });
      }

      // Award base points
      const basePoints = 100;
      gamification.points += basePoints;
      gamification.xp += basePoints;
      
      // Update stats
      gamification.stats.tasksCompleted += 1;
      gamification.stats.totalTimeSpent += (task.completionData?.actualDuration || 0);
      
      // Update efficiency average
      const completedTasks = gamification.stats.tasksCompleted;
      const currentAvg = gamification.stats.averageEfficiency;
      const newEfficiency = task.completionData?.efficiency || 0;
      gamification.stats.averageEfficiency = 
        ((currentAvg * (completedTasks - 1)) + newEfficiency) / completedTasks;

      // Update streaks
      await this.updateStreaks(gamification);

      await gamification.save();
    } catch (error) {
      logger.error('Error awarding task completion:', error);
    }
  },

  async updateStreaks(gamification) {
    const now = new Date();
    const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    const lastDate = gamification.streaks.daily.lastDate 
      ? new Date(gamification.streaks.daily.lastDate)
      : null;

    // Update daily streak
    if (!lastDate || lastDate.getTime() < today.getTime()) {
      const yesterday = new Date(today);
      yesterday.setDate(yesterday.getDate() - 1);

      if (lastDate && lastDate.getTime() === yesterday.getTime()) {
        // Continue streak
        gamification.streaks.daily.current += 1;
      } else {
        // New streak
        gamification.streaks.daily.current = 1;
      }

      if (gamification.streaks.daily.current > gamification.streaks.daily.longest) {
        gamification.streaks.daily.longest = gamification.streaks.daily.current;
      }

      gamification.streaks.daily.lastDate = today;
    }

    // Update weekly streak (simplified)
    const weekNumber = this.getWeekNumber(now);
    const lastWeek = gamification.streaks.weekly.lastWeek;

    if (!lastWeek || lastWeek !== weekNumber) {
      if (lastWeek && this.isConsecutiveWeek(lastWeek, weekNumber)) {
        gamification.streaks.weekly.current += 1;
      } else {
        gamification.streaks.weekly.current = 1;
      }

      if (gamification.streaks.weekly.current > gamification.streaks.weekly.longest) {
        gamification.streaks.weekly.longest = gamification.streaks.weekly.current;
      }

      gamification.streaks.weekly.lastWeek = weekNumber;
    }
  },

  getWeekNumber(date) {
    const d = new Date(Date.UTC(date.getFullYear(), date.getMonth(), date.getDate()));
    const dayNum = d.getUTCDay() || 7;
    d.setUTCDate(d.getUTCDate() + 4 - dayNum);
    const yearStart = new Date(Date.UTC(d.getUTCFullYear(), 0, 1));
    return d.getUTCFullYear() + '-' + Math.ceil((((d - yearStart) / 86400000) + 1) / 7);
  },

  isConsecutiveWeek(lastWeek, currentWeek) {
    // Simplified check - in production, handle year boundaries
    return lastWeek < currentWeek;
  }
};

module.exports = taskService;

