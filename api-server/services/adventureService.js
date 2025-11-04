const Adventure = require('../models/Adventure');
const User = require('../models/User');
const Event = require('../models/Event');
const Notification = require('../models/Notification');
const aiOrchestrator = require('./aiOrchestrator');
const externalApiService = require('./externalApiService');
const cacheService = require('./cacheService');
const logger = require('../utils/logger');

class AdventureService {
  async generateAdventure(userId, requestData) {
    try {
      const startTime = Date.now();
      
      // Get user preferences
      const user = await User.findById(userId);
      if (!user) {
        throw new Error('User not found');
      }

      // Check cache first
      const cacheKey = `adventure:${userId}:${requestData.location.lat}:${requestData.location.lng}:${requestData.interests.join(',')}`;
      const cachedAdventure = await cacheService.get(cacheKey);
      if (cachedAdventure) {
        logger.info(`Returning cached adventure for user ${userId}`);
        return cachedAdventure;
      }

      // Validate request data
      this.validateAdventureRequest(requestData);

      // Get external data
      const externalData = await externalApiService.getAdventureData(
        requestData.location,
        requestData.maxDistance || user.preferences.maxDistance,
        requestData.interests || user.preferences.interests
      );

      // Prepare AI input
      const aiInput = {
        userPreferences: user.preferences,
        location: requestData.location,
        nearbyEvents: externalData.events,
        weather: {
          current: externalData.weather,
          forecast: await externalApiService.getWeatherForecast(requestData.location, 1)
        },
        constraints: {
          duration: requestData.duration || user.preferences.preferredDuration,
          maxDistance: requestData.maxDistance || user.preferences.maxDistance,
          startTime: requestData.startTime || new Date(),
          endTime: requestData.endTime || new Date(Date.now() + (requestData.duration || user.preferences.preferredDuration) * 60000)
        }
      };

      // Generate adventure using AI
      const adventureData = await aiOrchestrator.generateAdventure(
        aiInput.userPreferences,
        aiInput.location,
        aiInput.nearbyEvents,
        aiInput.weather,
        aiInput.constraints
      );

      // Create adventure document
      const adventure = new Adventure({
        userId: userId,
        name: adventureData.adventure_name,
        description: adventureData.description,
        status: 'generated',
        steps: adventureData.steps.map(step => ({
          type: step.type,
          name: step.name,
          description: step.description,
          location: step.location,
          startTime: new Date(step.start_time),
          endTime: new Date(step.end_time),
          duration: step.duration,
          travelMethod: step.travel_method,
          travelDuration: step.travel_duration,
          travelDistance: step.travel_distance,
          task: step.task,
          venue: step.venue,
          weather: step.weather
        })),
        totalDuration: adventureData.total_duration,
        totalDistance: adventureData.total_distance,
        startLocation: requestData.location,
        endLocation: adventureData.steps[adventureData.steps.length - 1]?.location || requestData.location,
        preferences: {
          interests: requestData.interests || user.preferences.interests,
          skillLevel: user.preferences.skillLevel,
          socialMode: requestData.socialMode || user.preferences.socialMode,
          budget: user.preferences.budget,
          maxDistance: requestData.maxDistance || user.preferences.maxDistance
        },
        social: adventureData.social,
        weather: {
          forecast: aiInput.weather.forecast,
          alerts: adventureData.weather_alerts || []
        },
        aiMetadata: adventureData.aiMetadata,
        gamification: adventureData.gamification
      });

      await adventure.save();

      // Cache the adventure
      await cacheService.setAdventure(userId, requestData.location, requestData.interests, adventure);

      // Schedule notifications
      await this.scheduleAdventureNotifications(adventure);

      // Update user stats
      await user.updateStats(adventure);

      const generationTime = Date.now() - startTime;
      logger.info(`Adventure generated for user ${userId} in ${generationTime}ms`);

      return adventure;

    } catch (error) {
      logger.error('Adventure generation failed:', error);
      throw new Error(`Failed to generate adventure: ${error.message}`);
    }
  }

  async getAdventure(adventureId, userId) {
    try {
      const adventure = await Adventure.findOne({
        _id: adventureId,
        $or: [
          { userId: userId },
          { 'social.friendsInvited': userId },
          { 'social.isPublic': true }
        ]
      }).populate('userId', 'name profilePicture');

      if (!adventure) {
        throw new Error('Adventure not found');
      }

      return adventure;
    } catch (error) {
      logger.error('Failed to get adventure:', error);
      throw new Error(`Failed to get adventure: ${error.message}`);
    }
  }

  async getUserAdventures(userId, status = null, limit = 20, offset = 0) {
    try {
      const query = { userId: userId };
      if (status) {
        query.status = status;
      }

      const adventures = await Adventure.find(query)
        .sort({ 'metadata.generatedAt': -1 })
        .limit(limit)
        .skip(offset)
        .populate('userId', 'name profilePicture');

      return adventures;
    } catch (error) {
      logger.error('Failed to get user adventures:', error);
      throw new Error(`Failed to get user adventures: ${error.message}`);
    }
  }

  async startAdventure(adventureId, userId) {
    try {
      const adventure = await Adventure.findOne({
        _id: adventureId,
        userId: userId
      });

      if (!adventure) {
        throw new Error('Adventure not found');
      }

      if (adventure.status !== 'generated') {
        throw new Error('Adventure cannot be started');
      }

      adventure.status = 'active';
      adventure.metadata.startedAt = new Date();
      adventure.metadata.lastUpdated = new Date();

      await adventure.save();

      // Send real-time notification
      if (global.io) {
        global.io.to(`user_${userId}`).emit('adventure_started', {
          adventureId: adventure._id,
          name: adventure.name,
          message: 'Your adventure has started!'
        });
      }

      logger.info(`Adventure ${adventureId} started by user ${userId}`);
      return adventure;

    } catch (error) {
      logger.error('Failed to start adventure:', error);
      throw new Error(`Failed to start adventure: ${error.message}`);
    }
  }

  async completeAdventure(adventureId, userId, feedback = null) {
    try {
      const adventure = await Adventure.findOne({
        _id: adventureId,
        userId: userId
      });

      if (!adventure) {
        throw new Error('Adventure not found');
      }

      if (adventure.status !== 'active') {
        throw new Error('Adventure is not active');
      }

      adventure.status = 'completed';
      adventure.metadata.completedAt = new Date();
      adventure.metadata.lastUpdated = new Date();

      if (feedback) {
        adventure.feedback = {
          rating: feedback.rating,
          comments: feedback.comments,
          completedSteps: feedback.completedSteps || [],
          skippedSteps: feedback.skippedSteps || [],
          suggestions: feedback.suggestions,
          submittedAt: new Date()
        };
      }

      await adventure.save();

      // Update user stats
      const user = await User.findById(userId);
      if (user) {
        await user.updateStats(adventure);
      }

      // Send completion notification
      if (global.io) {
        global.io.to(`user_${userId}`).emit('adventure_completed', {
          adventureId: adventure._id,
          name: adventure.name,
          points: adventure.gamification.points,
          message: 'Adventure completed! Great job!'
        });
      }

      logger.info(`Adventure ${adventureId} completed by user ${userId}`);
      return adventure;

    } catch (error) {
      logger.error('Failed to complete adventure:', error);
      throw new Error(`Failed to complete adventure: ${error.message}`);
    }
  }

  async updateAdventureStep(adventureId, stepIndex, userId, action) {
    try {
      const adventure = await Adventure.findOne({
        _id: adventureId,
        userId: userId
      });

      if (!adventure) {
        throw new Error('Adventure not found');
      }

      if (adventure.status !== 'active') {
        throw new Error('Adventure is not active');
      }

      const step = adventure.steps[stepIndex];
      if (!step) {
        throw new Error('Step not found');
      }

      switch (action) {
        case 'complete':
          step.task.completed = true;
          break;
        case 'skip':
          step.task.skipped = true;
          break;
        case 'start':
          step.task.started = true;
          step.task.startedAt = new Date();
          break;
        default:
          throw new Error('Invalid action');
      }

      adventure.metadata.lastUpdated = new Date();
      await adventure.save();

      // Send real-time update
      if (global.io) {
        global.io.to(`adventure_${adventureId}`).emit('step_updated', {
          stepIndex,
          action,
          step: step
        });
      }

      return adventure;

    } catch (error) {
      logger.error('Failed to update adventure step:', error);
      throw new Error(`Failed to update adventure step: ${error.message}`);
    }
  }

  async getAdventureRecommendations(userId, location, limit = 5) {
    try {
      const user = await User.findById(userId);
      if (!user) {
        throw new Error('User not found');
      }

      // Find similar adventures by other users
      const similarAdventures = await Adventure.find({
        userId: { $ne: userId },
        'preferences.interests': { $in: user.preferences.interests },
        'startLocation.lat': {
          $gte: location.lat - 0.01,
          $lte: location.lat + 0.01
        },
        'startLocation.lng': {
          $gte: location.lng - 0.01,
          $lte: location.lng + 0.01
        },
        status: 'completed',
        'feedback.rating': { $gte: 4 }
      })
      .populate('userId', 'name profilePicture')
      .sort({ 'feedback.rating': -1, 'metadata.completedAt': -1 })
      .limit(limit);

      return similarAdventures;

    } catch (error) {
      logger.error('Failed to get adventure recommendations:', error);
      throw new Error(`Failed to get adventure recommendations: ${error.message}`);
    }
  }

  async shareAdventure(adventureId, userId, shareData) {
    try {
      const adventure = await Adventure.findOne({
        _id: adventureId,
        userId: userId
      });

      if (!adventure) {
        throw new Error('Adventure not found');
      }

      // Update social settings
      if (shareData.friends) {
        adventure.social.friendsInvited = shareData.friends;
      }
      if (shareData.isPublic !== undefined) {
        adventure.social.isPublic = shareData.isPublic;
      }

      await adventure.save();

      // Send invitations to friends
      if (shareData.friends && shareData.friends.length > 0) {
        for (const friendId of shareData.friends) {
          await Notification.createFriendNotification(
            friendId,
            'friend_invited',
            userId,
            `${user.name} invited you to join their adventure: ${adventure.name}`
          );
        }
      }

      return adventure;

    } catch (error) {
      logger.error('Failed to share adventure:', error);
      throw new Error(`Failed to share adventure: ${error.message}`);
    }
  }

  async scheduleAdventureNotifications(adventure) {
    try {
      const user = await User.findById(adventure.userId);
      if (!user) return;

      // Schedule step reminders
      for (let i = 0; i < adventure.steps.length; i++) {
        const step = adventure.steps[i];
        const reminderTime = new Date(step.startTime.getTime() - 15 * 60000); // 15 minutes before

        if (reminderTime > new Date()) {
          await Notification.createAdventureNotification(
            adventure.userId,
            'step_reminder',
            adventure._id,
            `Upcoming: ${step.name} in 15 minutes`,
            reminderTime
          );
        }
      }

      // Schedule completion reminder
      const completionTime = new Date(adventure.steps[adventure.steps.length - 1].endTime);
      await Notification.createAdventureNotification(
        adventure.userId,
        'adventure_completed',
        adventure._id,
        'How was your adventure? Leave a review!',
        completionTime
      );

    } catch (error) {
      logger.error('Failed to schedule adventure notifications:', error);
    }
  }

  validateAdventureRequest(requestData) {
    const required = ['location', 'interests'];
    for (const field of required) {
      if (!requestData[field]) {
        throw new Error(`Missing required field: ${field}`);
      }
    }

    if (!requestData.location.lat || !requestData.location.lng) {
      throw new Error('Invalid location data');
    }

    if (!Array.isArray(requestData.interests) || requestData.interests.length === 0) {
      throw new Error('Interests must be a non-empty array');
    }

    if (requestData.duration && (requestData.duration < 30 || requestData.duration > 90)) {
      throw new Error('Duration must be between 30 and 90 minutes');
    }

    if (requestData.maxDistance && (requestData.maxDistance < 1000 || requestData.maxDistance > 20000)) {
      throw new Error('Max distance must be between 1000 and 20000 meters');
    }
  }

  async getAdventureAnalytics(userId, timeRange = '30d') {
    try {
      const user = await User.findById(userId);
      if (!user) {
        throw new Error('User not found');
      }

      const timeRangeMs = this.getTimeRangeMs(timeRange);
      const startDate = new Date(Date.now() - timeRangeMs);

      const adventures = await Adventure.find({
        userId: userId,
        'metadata.generatedAt': { $gte: startDate }
      });

      const analytics = {
        totalAdventures: adventures.length,
        completedAdventures: adventures.filter(a => a.status === 'completed').length,
        activeAdventures: adventures.filter(a => a.status === 'active').length,
        totalPoints: adventures.reduce((sum, a) => sum + (a.gamification.points || 0), 0),
        averageRating: this.calculateAverageRating(adventures),
        favoriteInterests: this.getFavoriteInterests(adventures),
        completionRate: this.calculateCompletionRate(adventures),
        averageDuration: this.calculateAverageDuration(adventures),
        timeDistribution: this.getTimeDistribution(adventures)
      };

      return analytics;

    } catch (error) {
      logger.error('Failed to get adventure analytics:', error);
      throw new Error(`Failed to get adventure analytics: ${error.message}`);
    }
  }

  getTimeRangeMs(timeRange) {
    const ranges = {
      '7d': 7 * 24 * 60 * 60 * 1000,
      '30d': 30 * 24 * 60 * 60 * 1000,
      '90d': 90 * 24 * 60 * 60 * 1000,
      '1y': 365 * 24 * 60 * 60 * 1000
    };
    return ranges[timeRange] || ranges['30d'];
  }

  calculateAverageRating(adventures) {
    const ratedAdventures = adventures.filter(a => a.feedback && a.feedback.rating);
    if (ratedAdventures.length === 0) return 0;
    
    const sum = ratedAdventures.reduce((total, a) => total + a.feedback.rating, 0);
    return sum / ratedAdventures.length;
  }

  getFavoriteInterests(adventures) {
    const interestCount = {};
    adventures.forEach(adventure => {
      adventure.preferences.interests.forEach(interest => {
        interestCount[interest] = (interestCount[interest] || 0) + 1;
      });
    });

    return Object.entries(interestCount)
      .sort(([,a], [,b]) => b - a)
      .slice(0, 5)
      .map(([interest, count]) => ({ interest, count }));
  }

  calculateCompletionRate(adventures) {
    if (adventures.length === 0) return 0;
    const completed = adventures.filter(a => a.status === 'completed').length;
    return (completed / adventures.length) * 100;
  }

  calculateAverageDuration(adventures) {
    if (adventures.length === 0) return 0;
    const totalDuration = adventures.reduce((sum, a) => sum + a.totalDuration, 0);
    return totalDuration / adventures.length;
  }

  getTimeDistribution(adventures) {
    const distribution = {
      morning: 0,
      afternoon: 0,
      evening: 0,
      night: 0
    };

    adventures.forEach(adventure => {
      const hour = new Date(adventure.metadata.generatedAt).getHours();
      if (hour >= 6 && hour < 12) distribution.morning++;
      else if (hour >= 12 && hour < 17) distribution.afternoon++;
      else if (hour >= 17 && hour < 22) distribution.evening++;
      else distribution.night++;
    });

    return distribution;
  }
}

module.exports = new AdventureService();
