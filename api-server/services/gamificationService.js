const Gamification = require('../models/Gamification');
const Badge = require('../models/Badge');
const logger = require('../utils/logger');

const gamificationService = {
  async getOrCreateProfile(userId) {
    try {
      let profile = await Gamification.findOne({ userId });
      
      if (!profile) {
        profile = new Gamification({ userId });
        await profile.save();
      }
      
      return profile;
    } catch (error) {
      logger.error('Error getting gamification profile:', error);
      throw error;
    }
  },

  async getLeaderboard(limit = 100, period = 'all') {
    try {
      let query = {};
      
      // For future: filter by period (daily, weekly, monthly)
      // For now, return all-time leaderboard

      const leaderboard = await Gamification.find(query)
        .populate('userId', 'name email')
        .sort({ points: -1 })
        .limit(limit)
        .select('userId points level stats.stasksCompleted stats.challengesCompleted streaks');

      // Update ranks
      leaderboard.forEach((entry, index) => {
        entry.stats.currentRank = index + 1;
      });

      return leaderboard;
    } catch (error) {
      logger.error('Error fetching leaderboard:', error);
      throw error;
    }
  },

  async getUserRank(userId) {
    try {
      const userProfile = await Gamification.findOne({ userId });
      if (!userProfile) {
        return null;
      }

      const higherScorers = await Gamification.countDocuments({
        points: { $gt: userProfile.points }
      });

      return higherScorers + 1;
    } catch (error) {
      logger.error('Error calculating user rank:', error);
      throw error;
    }
  },

  async awardBadge(userId, badgeId) {
    try {
      const badge = await Badge.findById(badgeId);
      if (!badge || !badge.isActive) {
        throw new Error('Badge not found or inactive');
      }

      const profile = await this.getOrCreateProfile(userId);

      // Check if user already has this badge
      const hasBadge = profile.badges.some(b => b.badgeId.toString() === badgeId.toString());
      if (hasBadge) {
        return profile; // Already has badge
      }

      // Award badge
      profile.badges.push({
        badgeId: badge._id,
        badgeName: badge.name,
        badgeIcon: badge.icon,
        earnedAt: new Date()
      });

      // Award badge points
      profile.points += badge.pointsReward;
      profile.xp += badge.pointsReward;

      await profile.save();

      logger.info(`Badge awarded: ${badge.name} to user: ${userId}`);
      return profile;
    } catch (error) {
      logger.error('Error awarding badge:', error);
      throw error;
    }
  },

  async checkAndAwardBadges(userId) {
    try {
      const profile = await this.getOrCreateProfile(userId);
      const badges = await Badge.find({ isActive: true });

      const awardedBadges = [];

      for (const badge of badges) {
        // Check if user already has this badge
        const hasBadge = profile.badges.some(b => b.badgeId.toString() === badge._id.toString());
        if (hasBadge) continue;

        let shouldAward = false;

        switch (badge.criteria.type) {
          case 'streak':
            if (badge.criteria.value <= profile.streaks.daily.current) {
              shouldAward = true;
            }
            break;
          case 'tasks_completed':
            if (badge.criteria.value <= profile.stats.tasksCompleted) {
              shouldAward = true;
            }
            break;
          case 'challenges_completed':
            if (badge.criteria.value <= profile.stats.challengesCompleted) {
              shouldAward = true;
            }
            break;
          case 'points':
            if (badge.criteria.value <= profile.points) {
              shouldAward = true;
            }
            break;
          // Add more criteria types as needed
        }

        if (shouldAward) {
          await this.awardBadge(userId, badge._id);
          awardedBadges.push(badge);
        }
      }

      return awardedBadges;
    } catch (error) {
      logger.error('Error checking badges:', error);
      throw error;
    }
  },

  async updatePreferences(userId, preferences) {
    try {
      const profile = await this.getOrCreateProfile(userId);
      
      profile.preferences = {
        ...profile.preferences,
        ...preferences
      };

      await profile.save();
      return profile;
    } catch (error) {
      logger.error('Error updating preferences:', error);
      throw error;
    }
  }
};

module.exports = gamificationService;

