const User = require('../models/User');
const Adventure = require('../models/Adventure');
const Event = require('../models/Event');
const Notification = require('../models/Notification');
const cacheService = require('./cacheService');
const logger = require('../utils/logger');

class UserService {
  async createUser(userData) {
    try {
      // Check if user already exists
      const existingUser = await User.findOne({ email: userData.email });
      if (existingUser) {
        throw new Error('User already exists with this email');
      }

      // Create new user
      const user = new User({
        name: userData.name,
        email: userData.email,
        password: userData.password,
        preferences: userData.preferences || {
          interests: ['social'],
          skillLevel: 'beginner',
          maxDistance: 5000,
          preferredDuration: 60,
          socialMode: 'solo',
          budget: 'medium',
          timePreferences: {
            morning: false,
            afternoon: true,
            evening: true,
            night: false
          }
        }
      });

      await user.save();

      // Cache user preferences
      await cacheService.setUserPreferences(user._id, user.preferences);

      logger.info(`New user created: ${user.email}`);
      return user;

    } catch (error) {
      logger.error('Failed to create user:', error);
      throw new Error(`Failed to create user: ${error.message}`);
    }
  }

  async findByRefreshToken(token) {
    const user = await User.findOne({ 'refreshTokens.token': token, 'refreshTokens.expiresAt': { $gt: new Date() } });
    return user;
  }

  async revokeRefreshToken(token) {
    const user = await User.findOne({ 'refreshTokens.token': token });
    if (!user) return false;
    user.refreshTokens = user.refreshTokens.filter(rt => rt.token !== token);
    await user.save();
    return true;
  }

  async getUserById(userId) {
    try {
      // Check cache first
      const cached = await cacheService.get(`user:${userId}`);
      if (cached) {
        return cached;
      }

      const user = await User.findById(userId);
      if (!user) {
        throw new Error('User not found');
      }

      // Cache user data
      await cacheService.set(`user:${userId}`, user, 3600);

      return user;
    } catch (error) {
      logger.error('Failed to get user:', error);
      throw new Error(`Failed to get user: ${error.message}`);
    }
  }

  async getUserByEmail(email) {
    try {
      const user = await User.findOne({ email: email.toLowerCase() });
      if (!user) {
        throw new Error('User not found');
      }

      return user;
    } catch (error) {
      logger.error('Failed to get user by email:', error);
      throw new Error(`Failed to get user: ${error.message}`);
    }
  }

  async updateUser(userId, updateData) {
    try {
      const user = await User.findById(userId);
      if (!user) {
        throw new Error('User not found');
      }

      // Update allowed fields
      const allowedFields = ['name', 'preferences', 'location', 'profilePicture', 'bio'];
      for (const field of allowedFields) {
        if (updateData[field] !== undefined) {
          user[field] = updateData[field];
        }
      }

      user.updatedAt = new Date();
      await user.save();

      // Clear user cache
      await cacheService.clearUserCache(userId);

      logger.info(`User ${userId} updated`);
      return user;

    } catch (error) {
      logger.error('Failed to update user:', error);
      throw new Error(`Failed to update user: ${error.message}`);
    }
  }

  async updateUserPreferences(userId, preferences) {
    try {
      const user = await User.findById(userId);
      if (!user) {
        throw new Error('User not found');
      }

      // Validate preferences
      this.validatePreferences(preferences);

      user.preferences = { ...user.preferences, ...preferences };
      user.updatedAt = new Date();
      await user.save();

      // Update cache
      await cacheService.setUserPreferences(userId, user.preferences);

      logger.info(`User ${userId} preferences updated`);
      return user;

    } catch (error) {
      logger.error('Failed to update user preferences:', error);
      throw new Error(`Failed to update user preferences: ${error.message}`);
    }
  }

  async updateUserLocation(userId, location) {
    try {
      const user = await User.findById(userId);
      if (!user) {
        throw new Error('User not found');
      }

      user.location = {
        lat: location.lat,
        lng: location.lng,
        address: location.address,
        lastUpdated: new Date()
      };

      await user.save();

      // Clear user cache
      await cacheService.clearUserCache(userId);

      logger.info(`User ${userId} location updated`);
      return user;

    } catch (error) {
      logger.error('Failed to update user location:', error);
      throw new Error(`Failed to update user location: ${error.message}`);
    }
  }

  async addFriend(userId, friendId) {
    try {
      const user = await User.findById(userId);
      const friend = await User.findById(friendId);

      if (!user || !friend) {
        throw new Error('User or friend not found');
      }

      if (user.friends.includes(friendId)) {
        throw new Error('User is already a friend');
      }

      if (userId === friendId) {
        throw new Error('Cannot add yourself as a friend');
      }

      // Add friend to both users
      user.friends.push(friendId);
      friend.friends.push(userId);

      await Promise.all([user.save(), friend.save()]);

      // Send notification to friend
      await Notification.createFriendNotification(
        friendId,
        'friend_invited',
        userId,
        `${user.name} added you as a friend!`
      );

      logger.info(`User ${userId} added friend ${friendId}`);
      return { user, friend };

    } catch (error) {
      logger.error('Failed to add friend:', error);
      throw new Error(`Failed to add friend: ${error.message}`);
    }
  }

  async removeFriend(userId, friendId) {
    try {
      const user = await User.findById(userId);
      const friend = await User.findById(friendId);

      if (!user || !friend) {
        throw new Error('User or friend not found');
      }

      // Remove friend from both users
      user.friends = user.friends.filter(id => id.toString() !== friendId.toString());
      friend.friends = friend.friends.filter(id => id.toString() !== userId.toString());

      await Promise.all([user.save(), friend.save()]);

      logger.info(`User ${userId} removed friend ${friendId}`);
      return { user, friend };

    } catch (error) {
      logger.error('Failed to remove friend:', error);
      throw new Error(`Failed to remove friend: ${error.message}`);
    }
  }

  async getFriends(userId) {
    try {
      const user = await User.findById(userId).populate('friends', 'name profilePicture bio stats');
      if (!user) {
        throw new Error('User not found');
      }

      return user.friends;
    } catch (error) {
      logger.error('Failed to get friends:', error);
      throw new Error(`Failed to get friends: ${error.message}`);
    }
  }

  async searchUsers(query, userId, limit = 20) {
    try {
      const users = await User.find({
        _id: { $ne: userId },
        $or: [
          { name: { $regex: query, $options: 'i' } },
          { email: { $regex: query, $options: 'i' } }
        ]
      })
      .select('name profilePicture bio stats')
      .limit(limit);

      return users;
    } catch (error) {
      logger.error('Failed to search users:', error);
      throw new Error(`Failed to search users: ${error.message}`);
    }
  }

  async getUserStats(userId) {
    try {
      const user = await User.findById(userId);
      if (!user) {
        throw new Error('User not found');
      }

      // Get additional stats from adventures and events
      const [adventureStats, eventStats] = await Promise.all([
        this.getAdventureStats(userId),
        this.getEventStats(userId)
      ]);

      const stats = {
        ...user.stats,
        adventureStats,
        eventStats,
        friendsCount: user.friends.length,
        favoriteVenuesCount: user.favoriteVenues.length,
        accountAge: Math.floor((Date.now() - user.createdAt.getTime()) / (1000 * 60 * 60 * 24))
      };

      return stats;

    } catch (error) {
      logger.error('Failed to get user stats:', error);
      throw new Error(`Failed to get user stats: ${error.message}`);
    }
  }

  async getAdventureStats(userId) {
    try {
      const adventures = await Adventure.find({ userId: userId });

      const stats = {
        total: adventures.length,
        completed: adventures.filter(a => a.status === 'completed').length,
        active: adventures.filter(a => a.status === 'active').length,
        totalPoints: adventures.reduce((sum, a) => sum + (a.gamification.points || 0), 0),
        averageRating: this.calculateAverageAdventureRating(adventures),
        favoriteInterests: this.getFavoriteAdventureInterests(adventures)
      };

      return stats;

    } catch (error) {
      logger.error('Failed to get adventure stats:', error);
      return {};
    }
  }

  async getEventStats(userId) {
    try {
      const hostedEvents = await Event.find({ 'host.userId': userId });
      const attendedEvents = await Event.find({ 'attendees.userId': userId });

      const stats = {
        hosted: hostedEvents.length,
        attended: attendedEvents.length,
        totalCapacity: hostedEvents.reduce((sum, e) => sum + e.capacity, 0),
        totalAttendees: attendedEvents.reduce((sum, e) => sum + e.attendeeCount, 0)
      };

      return stats;

    } catch (error) {
      logger.error('Failed to get event stats:', error);
      return {};
    }
  }

  async addFavoriteVenue(userId, venueData) {
    try {
      const user = await User.findById(userId);
      if (!user) {
        throw new Error('User not found');
      }

      // Check if venue already exists
      const existingVenue = user.favoriteVenues.find(
        venue => venue.venueId === venueData.venueId
      );

      if (existingVenue) {
        throw new Error('Venue already in favorites');
      }

      user.favoriteVenues.push(venueData);
      await user.save();

      logger.info(`User ${userId} added favorite venue: ${venueData.name}`);
      return user;

    } catch (error) {
      logger.error('Failed to add favorite venue:', error);
      throw new Error(`Failed to add favorite venue: ${error.message}`);
    }
  }

  async removeFavoriteVenue(userId, venueId) {
    try {
      const user = await User.findById(userId);
      if (!user) {
        throw new Error('User not found');
      }

      user.favoriteVenues = user.favoriteVenues.filter(
        venue => venue.venueId !== venueId
      );

      await user.save();

      logger.info(`User ${userId} removed favorite venue: ${venueId}`);
      return user;

    } catch (error) {
      logger.error('Failed to remove favorite venue:', error);
      throw new Error(`Failed to remove favorite venue: ${error.message}`);
    }
  }

  async getUserLeaderboard(timeRange = '30d', limit = 50) {
    try {
      const timeRangeMs = this.getTimeRangeMs(timeRange);
      const startDate = new Date(Date.now() - timeRangeMs);

      const users = await User.find({
        'stats.lastAdventureDate': { $gte: startDate }
      })
      .sort({ 'stats.totalPoints': -1 })
      .limit(limit)
      .select('name profilePicture stats');

      return users.map((user, index) => ({
        rank: index + 1,
        userId: user._id,
        name: user.name,
        profilePicture: user.profilePicture,
        points: user.stats.totalPoints,
        adventuresCompleted: user.stats.adventuresCompleted,
        streak: user.stats.streak
      }));

    } catch (error) {
      logger.error('Failed to get leaderboard:', error);
      throw new Error(`Failed to get leaderboard: ${error.message}`);
    }
  }

  async deleteUser(userId) {
    try {
      const user = await User.findById(userId);
      if (!user) {
        throw new Error('User not found');
      }

      // Delete user's adventures
      await Adventure.deleteMany({ userId: userId });

      // Delete user's events
      await Event.deleteMany({ 'host.userId': userId });

      // Remove user from friends lists
      await User.updateMany(
        { friends: userId },
        { $pull: { friends: userId } }
      );

      // Delete user's notifications
      await Notification.deleteMany({ userId: userId });

      // Delete user
      await User.findByIdAndDelete(userId);

      // Clear user cache
      await cacheService.clearUserCache(userId);

      logger.info(`User ${userId} deleted`);
      return true;

    } catch (error) {
      logger.error('Failed to delete user:', error);
      throw new Error(`Failed to delete user: ${error.message}`);
    }
  }

  validatePreferences(preferences) {
    const validInterests = ['bars', 'music', 'food', 'outdoors', 'art', 'sports', 'social', 'solo', 'nightlife', 'culture'];
    const validSkillLevels = ['beginner', 'intermediate', 'advanced'];
    const validSocialModes = ['solo', 'friends', 'meet_new_people'];
    const validBudgets = ['low', 'medium', 'high'];

    if (preferences.interests) {
      if (!Array.isArray(preferences.interests)) {
        throw new Error('Interests must be an array');
      }
      for (const interest of preferences.interests) {
        if (!validInterests.includes(interest)) {
          throw new Error(`Invalid interest: ${interest}`);
        }
      }
    }

    if (preferences.skillLevel && !validSkillLevels.includes(preferences.skillLevel)) {
      throw new Error(`Invalid skill level: ${preferences.skillLevel}`);
    }

    if (preferences.socialMode && !validSocialModes.includes(preferences.socialMode)) {
      throw new Error(`Invalid social mode: ${preferences.socialMode}`);
    }

    if (preferences.budget && !validBudgets.includes(preferences.budget)) {
      throw new Error(`Invalid budget: ${preferences.budget}`);
    }

    if (preferences.maxDistance && (preferences.maxDistance < 1000 || preferences.maxDistance > 20000)) {
      throw new Error('Max distance must be between 1000 and 20000 meters');
    }

    if (preferences.preferredDuration && (preferences.preferredDuration < 30 || preferences.preferredDuration > 90)) {
      throw new Error('Preferred duration must be between 30 and 90 minutes');
    }
  }

  calculateAverageAdventureRating(adventures) {
    const ratedAdventures = adventures.filter(a => a.feedback && a.feedback.rating);
    if (ratedAdventures.length === 0) return 0;
    
    const sum = ratedAdventures.reduce((total, a) => total + a.feedback.rating, 0);
    return sum / ratedAdventures.length;
  }

  getFavoriteAdventureInterests(adventures) {
    const interestCount = {};
    adventures.forEach(adventure => {
      adventure.preferences.interests.forEach(interest => {
        interestCount[interest] = (interestCount[interest] || 0) + 1;
      });
    });

    return Object.entries(interestCount)
      .sort(([,a], [,b]) => b - a)
      .slice(0, 3)
      .map(([interest, count]) => ({ interest, count }));
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
}

module.exports = new UserService();
