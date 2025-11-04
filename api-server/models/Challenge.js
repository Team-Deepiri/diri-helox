const mongoose = require('mongoose');

const challengeSchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true,
    index: true
  },
  taskId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Task',
    required: true,
    index: true
  },
  type: {
    type: String,
    enum: ['quiz', 'puzzle', 'coding_challenge', 'timed_completion', 'streak', 'multiplayer', 'custom'],
    required: true
  },
  title: {
    type: String,
    required: true,
    trim: true,
    maxlength: 200
  },
  description: {
    type: String,
    trim: true,
    maxlength: 2000
  },
  difficulty: {
    type: String,
    enum: ['easy', 'medium', 'hard', 'adaptive'],
    default: 'medium'
  },
  difficultyScore: {
    type: Number,
    min: 1,
    max: 10,
    default: 5
  },
  status: {
    type: String,
    enum: ['pending', 'active', 'completed', 'failed', 'expired'],
    default: 'pending'
  },
  configuration: {
    // Challenge-specific settings
    timeLimit: Number, // in minutes
    attempts: Number,
    hints: [String],
    questions: [{
      question: String,
      options: [String],
      correctAnswer: Number,
      points: Number
    }],
    codeTemplate: String,
    testCases: [String],
    puzzleData: mongoose.Schema.Types.Mixed
  },
  pointsReward: {
    type: Number,
    default: 100,
    min: 0
  },
  bonusMultiplier: {
    type: Number,
    default: 1.0,
    min: 1.0,
    max: 3.0
  },
  completionData: {
    completedAt: Date,
    completionTime: Number, // in minutes
    score: Number,
    accuracy: Number, // percentage
    attemptsUsed: Number,
    hintsUsed: Number
  },
  aiGenerated: {
    type: Boolean,
    default: true
  },
  aiMetadata: {
    model: String,
    prompt: String,
    generationTime: Number
  },
  createdAt: {
    type: Date,
    default: Date.now
  },
  expiresAt: {
    type: Date
  },
  updatedAt: {
    type: Date,
    default: Date.now
  }
});

challengeSchema.index({ userId: 1, status: 1 });
challengeSchema.index({ userId: 1, type: 1 });
challengeSchema.index({ taskId: 1 });
challengeSchema.index({ createdAt: -1 });
challengeSchema.index({ expiresAt: 1 });

challengeSchema.pre('save', function(next) {
  this.updatedAt = Date.now();
  next();
});

module.exports = mongoose.model('Challenge', challengeSchema);

