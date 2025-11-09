/**
 * ELO Leaderboard Service
 * Implements ELO-like ranking system for competitive features
 */
const mongoose = require('mongoose');
const logger = require('../../utils/logger');

const ELORatingSchema = new mongoose.Schema({
  userId: { type: mongoose.Schema.Types.ObjectId, required: true, unique: true, index: true },
  rating: { type: Number, default: 1500, index: true },
  peakRating: { type: Number, default: 1500 },
  wins: { type: Number, default: 0 },
  losses: { type: Number, default: 0 },
  draws: { type: Number, default: 0 },
  totalMatches: { type: Number, default: 0 },
  winStreak: { type: Number, default: 0 },
  bestWinStreak: { type: Number, default: 0 },
  category: { type: String, default: 'overall', index: true },
  lastUpdated: { type: Date, default: Date.now }
}, {
  timestamps: true
});

ELORatingSchema.index({ category: 1, rating: -1 });
ELORatingSchema.index({ userId: 1, category: 1 }, { unique: true });

const ELORating = mongoose.model('ELORating', ELORatingSchema);

class ELOLeaderboardService {
  constructor() {
    this.K_FACTOR = 32; // Standard K-factor for ELO
    this.INITIAL_RATING = 1500;
    this.CATEGORIES = ['overall', 'coding', 'creative', 'study', 'productivity', 'social'];
  }

  /**
   * Get or create ELO rating
   */
  async getOrCreateRating(userId, category = 'overall') {
    try {
      let rating = await ELORating.findOne({ userId, category });
      
      if (!rating) {
        rating = new ELORating({
          userId,
          category,
          rating: this.INITIAL_RATING,
          peakRating: this.INITIAL_RATING
        });
        await rating.save();
      }
      
      return rating;
    } catch (error) {
      logger.error('Error getting ELO rating:', error);
      throw error;
    }
  }

  /**
   * Calculate expected score
   */
  _calculateExpectedScore(ratingA, ratingB) {
    return 1 / (1 + Math.pow(10, (ratingB - ratingA) / 400));
  }

  /**
   * Update ratings after match
   */
  async updateRatings(userId1, userId2, result, category = 'overall') {
    try {
      // result: 'win', 'loss', 'draw', or 1.0, 0.5, 0.0
      const rating1 = await this.getOrCreateRating(userId1, category);
      const rating2 = await this.getOrCreateRating(userId2, category);

      const expectedScore1 = this._calculateExpectedScore(rating1.rating, rating2.rating);
      const expectedScore2 = this._calculateExpectedScore(rating2.rating, rating1.rating);

      // Convert result to score
      let score1, score2;
      if (result === 'win' || result === 1.0) {
        score1 = 1.0;
        score2 = 0.0;
      } else if (result === 'loss' || result === 0.0) {
        score1 = 0.0;
        score2 = 1.0;
      } else {
        score1 = 0.5;
        score2 = 0.5;
      }

      // Calculate new ratings
      const newRating1 = rating1.rating + this.K_FACTOR * (score1 - expectedScore1);
      const newRating2 = rating2.rating + this.K_FACTOR * (score2 - expectedScore2);

      // Update rating 1
      rating1.rating = Math.round(newRating1);
      rating1.peakRating = Math.max(rating1.peakRating, rating1.rating);
      rating1.totalMatches += 1;
      
      if (score1 > score2) {
        rating1.wins += 1;
        rating1.winStreak += 1;
        rating1.bestWinStreak = Math.max(rating1.bestWinStreak, rating1.winStreak);
      } else if (score1 < score2) {
        rating1.losses += 1;
        rating1.winStreak = 0;
      } else {
        rating1.draws += 1;
      }

      // Update rating 2
      rating2.rating = Math.round(newRating2);
      rating2.peakRating = Math.max(rating2.peakRating, rating2.rating);
      rating2.totalMatches += 1;
      
      if (score2 > score1) {
        rating2.wins += 1;
        rating2.winStreak += 1;
        rating2.bestWinStreak = Math.max(rating2.bestWinStreak, rating2.winStreak);
      } else if (score2 < score1) {
        rating2.losses += 1;
        rating2.winStreak = 0;
      } else {
        rating2.draws += 1;
      }

      rating1.lastUpdated = new Date();
      rating2.lastUpdated = new Date();

      await rating1.save();
      await rating2.save();

      logger.info('ELO ratings updated', {
        userId1,
        userId2,
        category,
        newRating1: rating1.rating,
        newRating2: rating2.rating
      });

      return {
        user1: {
          userId: userId1,
          oldRating: rating1.rating - Math.round(this.K_FACTOR * (score1 - expectedScore1)),
          newRating: rating1.rating,
          change: rating1.rating - (rating1.rating - Math.round(this.K_FACTOR * (score1 - expectedScore1)))
        },
        user2: {
          userId: userId2,
          oldRating: rating2.rating - Math.round(this.K_FACTOR * (score2 - expectedScore2)),
          newRating: rating2.rating,
          change: rating2.rating - (rating2.rating - Math.round(this.K_FACTOR * (score2 - expectedScore2)))
        }
      };
    } catch (error) {
      logger.error('Error updating ELO ratings:', error);
      throw error;
    }
  }

  /**
   * Get leaderboard
   */
  async getLeaderboard(category = 'overall', limit = 100, offset = 0) {
    try {
      const ratings = await ELORating.find({ category })
        .populate('userId', 'name email avatar')
        .sort({ rating: -1 })
        .skip(offset)
        .limit(limit)
        .select('userId rating peakRating wins losses draws totalMatches winStreak bestWinStreak');

      const leaderboard = ratings.map((rating, index) => ({
        rank: offset + index + 1,
        user: rating.userId,
        rating: rating.rating,
        peakRating: rating.peakRating,
        wins: rating.wins,
        losses: rating.losses,
        draws: rating.draws,
        winRate: rating.totalMatches > 0 ? (rating.wins / rating.totalMatches) * 100 : 0,
        winStreak: rating.winStreak,
        bestWinStreak: rating.bestWinStreak
      }));

      return leaderboard;
    } catch (error) {
      logger.error('Error getting leaderboard:', error);
      throw error;
    }
  }

  /**
   * Get user rank
   */
  async getUserRank(userId, category = 'overall') {
    try {
      const userRating = await this.getOrCreateRating(userId, category);
      
      const rank = await ELORating.countDocuments({
        category,
        rating: { $gt: userRating.rating }
      }) + 1;

      const totalUsers = await ELORating.countDocuments({ category });

      return {
        rank,
        totalUsers,
        percentile: totalUsers > 0 ? ((totalUsers - rank) / totalUsers) * 100 : 0,
        rating: userRating.rating,
        peakRating: userRating.peakRating
      };
    } catch (error) {
      logger.error('Error getting user rank:', error);
      throw error;
    }
  }

  /**
   * Get rating history (simplified - would need separate collection for full history)
   */
  async getRatingInfo(userId, category = 'overall') {
    try {
      const rating = await this.getOrCreateRating(userId, category);
      
      return {
        rating: rating.rating,
        peakRating: rating.peakRating,
        wins: rating.wins,
        losses: rating.losses,
        draws: rating.draws,
        totalMatches: rating.totalMatches,
        winRate: rating.totalMatches > 0 ? (rating.wins / rating.totalMatches) * 100 : 0,
        winStreak: rating.winStreak,
        bestWinStreak: rating.bestWinStreak,
        category
      };
    } catch (error) {
      logger.error('Error getting rating info:', error);
      throw error;
    }
  }

  /**
   * Get top players
   */
  async getTopPlayers(category = 'overall', limit = 10) {
    try {
      return await this.getLeaderboard(category, limit, 0);
    } catch (error) {
      logger.error('Error getting top players:', error);
      throw error;
    }
  }

  /**
   * Find match (skill-based matchmaking)
   */
  async findMatch(userId, category = 'overall', ratingTolerance = 200) {
    try {
      const userRating = await this.getOrCreateRating(userId, category);
      
      const minRating = userRating.rating - ratingTolerance;
      const maxRating = userRating.rating + ratingTolerance;

      const potentialMatches = await ELORating.find({
        category,
        userId: { $ne: userId },
        rating: { $gte: minRating, $lte: maxRating }
      })
        .populate('userId', 'name email avatar')
        .limit(10)
        .select('userId rating');

      // Return random match from potential matches
      if (potentialMatches.length > 0) {
        const randomIndex = Math.floor(Math.random() * potentialMatches.length);
        return potentialMatches[randomIndex];
      }

      // If no matches in tolerance, expand search
      const expandedMatches = await ELORating.find({
        category,
        userId: { $ne: userId }
      })
        .populate('userId', 'name email avatar')
        .sort({ rating: 1 })
        .limit(5)
        .select('userId rating');

      return expandedMatches[0] || null;
    } catch (error) {
      logger.error('Error finding match:', error);
      throw error;
    }
  }

  /**
   * Reset rating (for testing/admin)
   */
  async resetRating(userId, category = 'overall') {
    try {
      const rating = await ELORating.findOne({ userId, category });
      if (rating) {
        rating.rating = this.INITIAL_RATING;
        rating.wins = 0;
        rating.losses = 0;
        rating.draws = 0;
        rating.totalMatches = 0;
        rating.winStreak = 0;
        await rating.save();
      }
      return rating;
    } catch (error) {
      logger.error('Error resetting rating:', error);
      throw error;
    }
  }
}

module.exports = new ELOLeaderboardService();

