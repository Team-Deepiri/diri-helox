const mongoose = require('mongoose');

const analyticsSchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true,
    index: true
  },
  date: {
    type: Date,
    required: true,
    index: true
  },
  metrics: {
    tasksCompleted: {
      type: Number,
      default: 0,
      min: 0
    },
    tasksCreated: {
      type: Number,
      default: 0,
      min: 0
    },
    challengesCompleted: {
      type: Number,
      default: 0,
      min: 0
    },
    timeSpent: {
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
    pointsEarned: {
      type: Number,
      default: 0,
      min: 0
    },
    badgesEarned: {
      type: Number,
      default: 0,
      min: 0
    },
    peakProductivityHour: {
      type: Number,
      min: 0,
      max: 23
    },
    tasksByType: {
      manual: { type: Number, default: 0 },
      notion: { type: Number, default: 0 },
      trello: { type: Number, default: 0 },
      github: { type: Number, default: 0 },
      google_docs: { type: Number, default: 0 },
      study: { type: Number, default: 0 },
      creative: { type: Number, default: 0 }
    },
    challengesByType: {
      quiz: { type: Number, default: 0 },
      puzzle: { type: Number, default: 0 },
      coding_challenge: { type: Number, default: 0 },
      timed_completion: { type: Number, default: 0 },
      streak: { type: Number, default: 0 },
      multiplayer: { type: Number, default: 0 }
    }
  },
  insights: [{
    type: {
      type: String,
      enum: ['efficiency_trend', 'peak_hours', 'task_type_preference', 'challenge_performance', 'streak_suggestion', 'break_timing'],
      required: true
    },
    message: String,
    data: mongoose.Schema.Types.Mixed,
    priority: {
      type: String,
      enum: ['low', 'medium', 'high'],
      default: 'medium'
    }
  }],
  createdAt: {
    type: Date,
    default: Date.now
  },
  updatedAt: {
    type: Date,
    default: Date.now
  }
});

analyticsSchema.index({ userId: 1, date: -1 });
analyticsSchema.index({ userId: 1, 'metrics.date': -1 });

analyticsSchema.pre('save', function(next) {
  this.updatedAt = Date.now();
  next();
});

module.exports = mongoose.model('Analytics', analyticsSchema);

