/**
 * Badge System Service
 * Manages 500+ unique badges with dynamic conditions
 */
const mongoose = require('mongoose');
const logger = require('../utils/logger');

const BadgeSchema = new mongoose.Schema({
  badgeId: { type: String, required: true, unique: true, index: true },
  name: { type: String, required: true },
  description: { type: String, required: true },
  category: {
    type: String,
    enum: ['productivity', 'skill', 'social', 'achievement', 'special', 'seasonal', 'secret'],
    required: true
  },
  rarity: {
    type: String,
    enum: ['common', 'uncommon', 'rare', 'epic', 'legendary', 'mythic'],
    default: 'common'
  },
  icon: { type: String }, // Emoji or icon URL
  conditions: mongoose.Schema.Types.Mixed, // Dynamic conditions
  isSecret: { type: Boolean, default: false },
  isProgressive: { type: Boolean, default: false },
  tiers: [{
    tier: Number,
    name: String,
    description: String,
    condition: mongoose.Schema.Types.Mixed
  }],
  unlockable: { type: Boolean, default: true },
  createdAt: { type: Date, default: Date.now }
}, {
  timestamps: true
});

const Badge = mongoose.model('Badge', BadgeSchema);

const UserBadgeSchema = new mongoose.Schema({
  userId: { type: mongoose.Schema.Types.ObjectId, required: true, index: true },
  badgeId: { type: String, required: true, index: true },
  unlockedAt: { type: Date, default: Date.now },
  progress: { type: Number, default: 0 },
  tier: { type: Number, default: 1 },
  metadata: mongoose.Schema.Types.Mixed
}, {
  timestamps: false
});

UserBadgeSchema.index({ userId: 1, badgeId: 1 }, { unique: true });

const UserBadge = mongoose.model('UserBadge', UserBadgeSchema);

class BadgeSystemService {
  constructor() {
    this.badgeDefinitions = this._loadBadgeDefinitions();
  }

  /**
   * Load badge definitions (500+ badges)
   */
  _loadBadgeDefinitions() {
    // This would typically load from a JSON file or database
    // For now, return structure
    return {
      // Productivity badges
      'first_task': {
        badgeId: 'first_task',
        name: 'First Steps',
        description: 'Complete your first task',
        category: 'productivity',
        rarity: 'common',
        icon: '🎯',
        conditions: { tasksCompleted: 1 }
      },
      // ... 500+ more badges would be defined here
    };
  }

  /**
   * Check and award badges
   */
  async checkAndAwardBadges(userId, eventType, eventData) {
    try {
      const awardedBadges = [];

      // Get all badges that might be relevant
      const relevantBadges = await Badge.find({
        $or: [
          { 'conditions.eventType': eventType },
          { 'conditions.any': true }
        ]
      });

      for (const badge of relevantBadges) {
        // Check if user already has this badge
        const existing = await UserBadge.findOne({ userId, badgeId: badge.badgeId });
        
        if (existing && !badge.isProgressive) {
          continue; // Already unlocked and not progressive
        }

        // Check if conditions are met
        if (await this._checkConditions(badge, userId, eventData)) {
          if (existing) {
            // Progressive badge - update tier
            const newTier = await this._getNextTier(badge, existing.tier, eventData);
            if (newTier > existing.tier) {
              existing.tier = newTier;
              existing.progress = 100;
              existing.unlockedAt = new Date();
              await existing.save();
              awardedBadges.push({ badge, tier: newTier, upgraded: true });
            }
          } else {
            // New badge unlock
            const userBadge = new UserBadge({
              userId,
              badgeId: badge.badgeId,
              tier: 1,
              progress: 100
            });
            await userBadge.save();
            awardedBadges.push({ badge, tier: 1, upgraded: false });
          }
        } else if (badge.isProgressive && existing) {
          // Update progress for progressive badges
          const progress = await this._calculateProgress(badge, userId, eventData);
          existing.progress = progress;
          await existing.save();
        }
      }

      if (awardedBadges.length > 0) {
        logger.info('Badges awarded', { userId, count: awardedBadges.length });
      }

      return awardedBadges;
    } catch (error) {
      logger.error('Error checking badges:', error);
      throw error;
    }
  }

  /**
   * Check if badge conditions are met
   */
  async _checkConditions(badge, userId, eventData) {
    try {
      const conditions = badge.conditions;
      
      // Simple condition checking (can be expanded)
      if (conditions.tasksCompleted) {
        const taskCount = await this._getTaskCount(userId);
        return taskCount >= conditions.tasksCompleted;
      }

      if (conditions.challengesCompleted) {
        const challengeCount = await this._getChallengeCount(userId);
        return challengeCount >= conditions.challengesCompleted;
      }

      if (conditions.streak) {
        const streak = await this._getStreak(userId);
        return streak >= conditions.streak;
      }

      if (conditions.skillLevel) {
        const skillLevel = await this._getSkillLevel(userId, conditions.skillLevel.skill);
        return skillLevel >= conditions.skillLevel.level;
      }

      // Add more condition types as needed

      return false;
    } catch (error) {
      logger.error('Error checking conditions:', error);
      return false;
    }
  }

  /**
   * Get user badges
   */
  async getUserBadges(userId, category = null) {
    try {
      const query = { userId };
      const userBadges = await UserBadge.find(query)
        .populate('badgeId')
        .sort({ unlockedAt: -1 });

      let badges = userBadges.map(ub => ({
        badge: ub.badgeId,
        unlockedAt: ub.unlockedAt,
        progress: ub.progress,
        tier: ub.tier,
        metadata: ub.metadata
      }));

      if (category) {
        badges = badges.filter(b => b.badge.category === category);
      }

      return badges;
    } catch (error) {
      logger.error('Error getting user badges:', error);
      throw error;
    }
  }

  /**
   * Get badge statistics
   */
  async getBadgeStats(userId) {
    try {
      const userBadges = await UserBadge.find({ userId });
      const allBadges = await Badge.countDocuments();

      const stats = {
        totalUnlocked: userBadges.length,
        totalAvailable: allBadges,
        completionRate: (userBadges.length / allBadges) * 100,
        byCategory: {},
        byRarity: {},
        progressive: userBadges.filter(b => b.progress < 100).length
      };

      // Get badge details for categorization
      const badgeIds = userBadges.map(b => b.badgeId);
      const badges = await Badge.find({ badgeId: { $in: badgeIds } });

      badges.forEach(badge => {
        // Count by category
        stats.byCategory[badge.category] = (stats.byCategory[badge.category] || 0) + 1;
        
        // Count by rarity
        stats.byRarity[badge.rarity] = (stats.byRarity[badge.rarity] || 0) + 1;
      });

      return stats;
    } catch (error) {
      logger.error('Error getting badge stats:', error);
      throw error;
    }
  }

  /**
   * Get available badges (not yet unlocked)
   */
  async getAvailableBadges(userId, category = null) {
    try {
      const userBadgeIds = (await UserBadge.find({ userId }).select('badgeId'))
        .map(b => b.badgeId);

      const query = { badgeId: { $nin: userBadgeIds }, unlockable: true };
      if (category) {
        query.category = category;
      }

      const badges = await Badge.find(query)
        .sort({ rarity: 1, createdAt: 1 });

      return badges;
    } catch (error) {
      logger.error('Error getting available badges:', error);
      throw error;
    }
  }

  /**
   * Get secret badges (discovery mechanics)
   */
  async discoverSecretBadge(userId, discoveryKey) {
    try {
      const secretBadges = await Badge.find({
        isSecret: true,
        'conditions.discoveryKey': discoveryKey
      });

      if (secretBadges.length === 0) {
        return null;
      }

      const badge = secretBadges[0];
      
      // Check if already unlocked
      const existing = await UserBadge.findOne({ userId, badgeId: badge.badgeId });
      if (existing) {
        return { badge, alreadyUnlocked: true };
      }

      // Unlock secret badge
      const userBadge = new UserBadge({
        userId,
        badgeId: badge.badgeId,
        tier: 1,
        progress: 100
      });
      await userBadge.save();

      logger.info('Secret badge discovered', { userId, badgeId: badge.badgeId });
      return { badge, alreadyUnlocked: false };
    } catch (error) {
      logger.error('Error discovering secret badge:', error);
      throw error;
    }
  }

  async _getTaskCount(userId) {
    // Placeholder - would query task service
    return 0;
  }

  async _getChallengeCount(userId) {
    // Placeholder - would query challenge service
    return 0;
  }

  async _getStreak(userId) {
    // Placeholder - would query gamification service
    return 0;
  }

  async _getSkillLevel(userId, skill) {
    // Placeholder - would query skill tree service
    return 1;
  }

  async _calculateProgress(badge, userId, eventData) {
    // Calculate progress percentage for progressive badges
    return 0;
  }

  async _getNextTier(badge, currentTier, eventData) {
    // Determine next tier for progressive badges
    return currentTier;
  }
}

module.exports = new BadgeSystemService();

