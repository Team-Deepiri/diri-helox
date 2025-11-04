const mongoose = require('mongoose');

const gamificationSchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true,
    unique: true,
    index: true
  },
  points: {
    type: Number,
    default: 0,
    min: 0
  },
  level: {
    type: Number,
    default: 1,
    min: 1
  },
  xp: {
    type: Number,
    default: 0,
    min: 0
  },
  xpToNextLevel: {
    type: Number,
    default: 1000
  },
  streaks: {
    daily: {
      current: {
        type: Number,
        default: 0,
        min: 0
      },
      longest: {
        type: Number,
        default: 0,
        min: 0
      },
      lastDate: Date
    },
    weekly: {
      current: {
        type: Number,
        default: 0,
        min: 0
      },
      longest: {
        type: Number,
        default: 0,
        min: 0
      },
      lastWeek: String // Format: "YYYY-WW"
    }
  },
  badges: [{
    badgeId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'Badge'
    },
    badgeName: String,
    badgeIcon: String,
    earnedAt: {
      type: Date,
      default: Date.now
    }
  }],
  achievements: [{
    achievementId: String,
    achievementName: String,
    unlockedAt: {
      type: Date,
      default: Date.now
    },
    progress: {
      type: Number,
      default: 100,
      min: 0,
      max: 100
    }
  }],
  stats: {
    tasksCompleted: {
      type: Number,
      default: 0,
      min: 0
    },
    challengesCompleted: {
      type: Number,
      default: 0,
      min: 0
    },
    totalTimeSpent: {
      type: Number,
      default: 0,
      min: 0
    }, // in minutes
    averageEfficiency: {
      type: Number,
      default: 0,
      min: 0,
      max: 100
    },
    perfectCompletions: {
      type: Number,
      default: 0,
      min: 0
    },
    currentRank: {
      type: Number,
      default: 0
    }
  },
  preferences: {
    showLeaderboard: {
      type: Boolean,
      default: true
    },
    enableNotifications: {
      type: Boolean,
      default: true
    },
    challengeDifficulty: {
      type: String,
      enum: ['easy', 'medium', 'hard', 'adaptive'],
      default: 'adaptive'
    }
  },
  createdAt: {
    type: Date,
    default: Date.now
  },
  updatedAt: {
    type: Date,
    default: Date.now
  }
});

gamificationSchema.index({ userId: 1 });
gamificationSchema.index({ points: -1 });
gamificationSchema.index({ 'stats.tasksCompleted': -1 });
gamificationSchema.index({ level: -1 });

gamificationSchema.pre('save', function(next) {
  this.updatedAt = Date.now();
  // Auto-level up if XP threshold reached
  if (this.xp >= this.xpToNextLevel) {
    this.level += 1;
    this.xp = this.xp - this.xpToNextLevel;
    this.xpToNextLevel = Math.floor(this.xpToNextLevel * 1.5); // Exponential leveling
  }
  next();
});

module.exports = mongoose.model('Gamification', gamificationSchema);

