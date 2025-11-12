/**
 * Multi-Currency Service
 * Manages multiple currency types (XP, coins, gems, etc.)
 */
const mongoose = require('mongoose');
const logger = require('../utils/logger');

const CurrencyBalanceSchema = new mongoose.Schema({
  userId: { type: mongoose.Schema.Types.ObjectId, required: true, unique: true, index: true },
  currencies: {
    xp: { type: Number, default: 0 },
    coins: { type: Number, default: 0 },
    gems: { type: Number, default: 0 },
    energy: { type: Number, default: 100, max: 100 },
    tokens: { type: Number, default: 0 },
    stars: { type: Number, default: 0 }
  },
  totalEarned: {
    xp: { type: Number, default: 0 },
    coins: { type: Number, default: 0 },
    gems: { type: Number, default: 0 }
  },
  lastUpdated: { type: Date, default: Date.now }
}, {
  timestamps: true
});

const CurrencyBalance = mongoose.model('CurrencyBalance', CurrencyBalanceSchema);

const CurrencyTransactionSchema = new mongoose.Schema({
  userId: { type: mongoose.Schema.Types.ObjectId, required: true, index: true },
  currencyType: {
    type: String,
    enum: ['xp', 'coins', 'gems', 'energy', 'tokens', 'stars'],
    required: true
  },
  amount: { type: Number, required: true },
  transactionType: {
    type: String,
    enum: ['earn', 'spend', 'reward', 'purchase', 'refund', 'transfer'],
    required: true
  },
  source: { type: String }, // challenge, task, achievement, etc.
  sourceId: { type: mongoose.Schema.Types.ObjectId },
  description: { type: String },
  metadata: mongoose.Schema.Types.Mixed,
  createdAt: { type: Date, default: Date.now, index: true }
}, {
  timestamps: false
});

CurrencyTransactionSchema.index({ userId: 1, createdAt: -1 });

const CurrencyTransaction = mongoose.model('CurrencyTransaction', CurrencyTransactionSchema);

class MultiCurrencyService {
  constructor() {
    this.CURRENCIES = ['xp', 'coins', 'gems', 'energy', 'tokens', 'stars'];
    this.MAX_ENERGY = 100;
    this.ENERGY_REGENERATION_RATE = 1; // per minute
  }

  /**
   * Get or create currency balance
   */
  async getOrCreateBalance(userId) {
    try {
      let balance = await CurrencyBalance.findOne({ userId });
      
      if (!balance) {
        balance = new CurrencyBalance({ userId });
        await balance.save();
      }
      
      return balance;
    } catch (error) {
      logger.error('Error getting currency balance:', error);
      throw error;
    }
  }

  /**
   * Award currency
   */
  async awardCurrency(userId, currencyType, amount, source = null, sourceId = null, description = null) {
    try {
      if (!this.CURRENCIES.includes(currencyType)) {
        throw new Error(`Invalid currency type: ${currencyType}`);
      }

      const balance = await this.getOrCreateBalance(userId);
      
      // Update balance
      balance.currencies[currencyType] = (balance.currencies[currencyType] || 0) + amount;
      
      // Handle energy cap
      if (currencyType === 'energy') {
        balance.currencies.energy = Math.min(balance.currencies.energy, this.MAX_ENERGY);
      }

      // Update total earned (only for positive amounts)
      if (amount > 0 && ['xp', 'coins', 'gems'].includes(currencyType)) {
        balance.totalEarned[currencyType] = (balance.totalEarned[currencyType] || 0) + amount;
      }

      balance.lastUpdated = new Date();
      await balance.save();

      // Record transaction
      const transaction = new CurrencyTransaction({
        userId,
        currencyType,
        amount,
        transactionType: amount > 0 ? 'earn' : 'spend',
        source,
        sourceId,
        description: description || `Awarded ${amount} ${currencyType}`
      });

      await transaction.save();

      logger.info('Currency awarded', { userId, currencyType, amount, source });
      
      return {
        currencyType,
        newBalance: balance.currencies[currencyType],
        totalEarned: balance.totalEarned[currencyType] || 0
      };
    } catch (error) {
      logger.error('Error awarding currency:', error);
      throw error;
    }
  }

  /**
   * Spend currency
   */
  async spendCurrency(userId, currencyType, amount, description = null) {
    try {
      const balance = await this.getOrCreateBalance(userId);
      
      if (balance.currencies[currencyType] < amount) {
        throw new Error(`Insufficient ${currencyType}. Current: ${balance.currencies[currencyType]}, Required: ${amount}`);
      }

      balance.currencies[currencyType] -= amount;
      balance.lastUpdated = new Date();
      await balance.save();

      // Record transaction
      const transaction = new CurrencyTransaction({
        userId,
        currencyType,
        amount: -amount,
        transactionType: 'spend',
        description: description || `Spent ${amount} ${currencyType}`
      });

      await transaction.save();

      logger.info('Currency spent', { userId, currencyType, amount });
      
      return {
        currencyType,
        newBalance: balance.currencies[currencyType],
        spent: amount
      };
    } catch (error) {
      logger.error('Error spending currency:', error);
      throw error;
    }
  }

  /**
   * Get balance
   */
  async getBalance(userId) {
    try {
      const balance = await this.getOrCreateBalance(userId);
      return balance.currencies;
    } catch (error) {
      logger.error('Error getting balance:', error);
      throw error;
    }
  }

  /**
   * Get transaction history
   */
  async getTransactionHistory(userId, currencyType = null, limit = 100) {
    try {
      const query = { userId };
      if (currencyType) {
        query.currencyType = currencyType;
      }

      const transactions = await CurrencyTransaction.find(query)
        .sort({ createdAt: -1 })
        .limit(limit)
        .select('currencyType amount transactionType source description createdAt');

      return transactions;
    } catch (error) {
      logger.error('Error getting transaction history:', error);
      throw error;
    }
  }

  /**
   * Regenerate energy
   */
  async regenerateEnergy(userId) {
    try {
      const balance = await this.getOrCreateBalance(userId);
      
      if (balance.currencies.energy >= this.MAX_ENERGY) {
        return { energy: balance.currencies.energy, regenerated: 0 };
      }

      const minutesSinceUpdate = (Date.now() - balance.lastUpdated) / (1000 * 60);
      const regenerated = Math.floor(minutesSinceUpdate * this.ENERGY_REGENERATION_RATE);
      
      if (regenerated > 0) {
        balance.currencies.energy = Math.min(
          balance.currencies.energy + regenerated,
          this.MAX_ENERGY
        );
        balance.lastUpdated = new Date();
        await balance.save();
      }

      return {
        energy: balance.currencies.energy,
        regenerated: Math.min(regenerated, this.MAX_ENERGY - balance.currencies.energy)
      };
    } catch (error) {
      logger.error('Error regenerating energy:', error);
      throw error;
    }
  }

  /**
   * Convert currency
   */
  async convertCurrency(userId, fromCurrency, toCurrency, amount, rate) {
    try {
      const fromAmount = amount;
      const toAmount = Math.floor(amount * rate);

      // Spend from currency
      await this.spendCurrency(userId, fromCurrency, fromAmount, `Converted to ${toCurrency}`);
      
      // Award to currency
      await this.awardCurrency(userId, toCurrency, toAmount, 'conversion', null, `Converted from ${fromCurrency}`);

      logger.info('Currency converted', { userId, fromCurrency, toCurrency, fromAmount, toAmount });
      
      return {
        fromCurrency,
        fromAmount,
        toCurrency,
        toAmount,
        rate
      };
    } catch (error) {
      logger.error('Error converting currency:', error);
      throw error;
    }
  }

  /**
   * Get currency statistics
   */
  async getCurrencyStats(userId) {
    try {
      const balance = await this.getOrCreateBalance(userId);
      
      const stats = await CurrencyTransaction.aggregate([
        { $match: { userId: new mongoose.Types.ObjectId(userId) } },
        {
          $group: {
            _id: '$currencyType',
            totalEarned: {
              $sum: { $cond: [{ $gt: ['$amount', 0] }, '$amount', 0] }
            },
            totalSpent: {
              $sum: { $cond: [{ $lt: ['$amount', 0] }, { $abs: '$amount' }, 0] }
            },
            transactionCount: { $sum: 1 }
          }
        }
      ]);

      const statsMap = {};
      stats.forEach(stat => {
        statsMap[stat._id] = {
          totalEarned: stat.totalEarned,
          totalSpent: stat.totalSpent,
          transactionCount: stat.transactionCount,
          currentBalance: balance.currencies[stat._id] || 0
        };
      });

      return statsMap;
    } catch (error) {
      logger.error('Error getting currency stats:', error);
      throw error;
    }
  }
}

module.exports = new MultiCurrencyService();

