const mongoose = require('mongoose');

const taskSchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true,
    index: true
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
  type: {
    type: String,
    enum: ['manual', 'notion', 'trello', 'github', 'google_docs', 'pdf', 'code', 'study', 'creative'],
    default: 'manual'
  },
  status: {
    type: String,
    enum: ['pending', 'in_progress', 'completed', 'cancelled'],
    default: 'pending'
  },
  priority: {
    type: String,
    enum: ['low', 'medium', 'high', 'urgent'],
    default: 'medium'
  },
  dueDate: {
    type: Date
  },
  estimatedDuration: {
    type: Number, // in minutes
    min: 1
  },
  tags: [{
    type: String,
    trim: true,
    maxlength: 50
  }],
  metadata: {
    // For integration-specific data
    sourceId: String,
    sourceUrl: String,
    sourceData: mongoose.Schema.Types.Mixed
  },
  challengeId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Challenge'
  },
  completionData: {
    completedAt: Date,
    actualDuration: Number, // in minutes
    efficiency: Number, // percentage (actualDuration / estimatedDuration * 100)
    notes: String
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

taskSchema.index({ userId: 1, status: 1 });
taskSchema.index({ userId: 1, type: 1 });
taskSchema.index({ userId: 1, dueDate: 1 });
taskSchema.index({ createdAt: -1 });

taskSchema.pre('save', function(next) {
  this.updatedAt = Date.now();
  next();
});

module.exports = mongoose.model('Task', taskSchema);

