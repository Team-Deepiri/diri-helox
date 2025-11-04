const mongoose = require('mongoose');

const notificationSchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  type: {
    type: String,
    enum: [
      'adventure_generated',
      'step_reminder',
      'weather_alert',
      'venue_change',
      'friend_joined',
      'friend_invited',
      'event_reminder',
      'event_cancelled',
      'event_updated',
      'badge_earned',
      'points_earned',
      'streak_reminder',
      'adventure_completed',
      'new_event_nearby',
      'friend_adventure_shared',
      'system_announcement'
    ],
    required: true
  },
  title: {
    type: String,
    required: true,
    maxlength: 100
  },
  message: {
    type: String,
    required: true,
    maxlength: 500
  },
  data: {
    adventureId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'Adventure'
    },
    eventId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'Event'
    },
    stepIndex: Number,
    points: Number,
    badge: String,
    friendId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'User'
    },
    metadata: mongoose.Schema.Types.Mixed
  },
  priority: {
    type: String,
    enum: ['low', 'medium', 'high', 'urgent'],
    default: 'medium'
  },
  channels: [{
    type: String,
    enum: ['push', 'email', 'sms', 'in_app'],
    default: ['push', 'in_app']
  }],
  status: {
    type: String,
    enum: ['pending', 'sent', 'delivered', 'read', 'failed'],
    default: 'pending'
  },
  scheduledFor: {
    type: Date,
    default: Date.now
  },
  sentAt: Date,
  deliveredAt: Date,
  readAt: Date,
  expiresAt: Date,
  retryCount: {
    type: Number,
    default: 0,
    max: 3
  },
  errorMessage: String,
  metadata: {
    source: String,
    campaign: String,
    version: { type: String, default: '2.0' }
  }
}, {
  timestamps: true
});

// Indexes for better query performance
notificationSchema.index({ userId: 1, status: 1 });
notificationSchema.index({ scheduledFor: 1 });
notificationSchema.index({ type: 1 });
notificationSchema.index({ 'data.adventureId': 1 });
notificationSchema.index({ 'data.eventId': 1 });

// Virtual for time until scheduled
notificationSchema.virtual('timeUntilScheduled').get(function() {
  if (!this.scheduledFor) return 0;
  return Math.max(0, this.scheduledFor.getTime() - Date.now());
});

// Virtual for is overdue
notificationSchema.virtual('isOverdue').get(function() {
  return this.scheduledFor && this.scheduledFor < new Date() && this.status === 'pending';
});

// Method to mark as sent
notificationSchema.methods.markAsSent = function() {
  this.status = 'sent';
  this.sentAt = new Date();
  return this.save();
};

// Method to mark as delivered
notificationSchema.methods.markAsDelivered = function() {
  this.status = 'delivered';
  this.deliveredAt = new Date();
  return this.save();
};

// Method to mark as read
notificationSchema.methods.markAsRead = function() {
  this.status = 'read';
  this.readAt = new Date();
  return this.save();
};

// Method to mark as failed
notificationSchema.methods.markAsFailed = function(errorMessage) {
  this.status = 'failed';
  this.errorMessage = errorMessage;
  this.retryCount += 1;
  return this.save();
};

// Method to reschedule
notificationSchema.methods.reschedule = function(newTime) {
  this.scheduledFor = newTime;
  this.status = 'pending';
  return this.save();
};

// Static method to find pending notifications
notificationSchema.statics.findPending = function() {
  return this.find({
    status: 'pending',
    scheduledFor: { $lte: new Date() },
    retryCount: { $lt: 3 }
  });
};

// Static method to find overdue notifications
notificationSchema.statics.findOverdue = function() {
  return this.find({
    status: 'pending',
    scheduledFor: { $lt: new Date() },
    retryCount: { $lt: 3 }
  });
};

// Static method to find notifications for user
notificationSchema.statics.findForUser = function(userId, limit = 50) {
  return this.find({ userId })
    .sort({ createdAt: -1 })
    .limit(limit)
    .populate('data.adventureId', 'name status')
    .populate('data.eventId', 'name startTime')
    .populate('data.friendId', 'name profilePicture');
};

// Static method to create adventure notification
notificationSchema.statics.createAdventureNotification = function(userId, type, adventureId, message, scheduledFor = null) {
  return this.create({
    userId,
    type,
    title: 'Adventure Update',
    message,
    data: { adventureId },
    scheduledFor: scheduledFor || new Date(),
    priority: type === 'weather_alert' ? 'high' : 'medium'
  });
};

// Static method to create event notification
notificationSchema.statics.createEventNotification = function(userId, type, eventId, message, scheduledFor = null) {
  return this.create({
    userId,
    type,
    title: 'Event Update',
    message,
    data: { eventId },
    scheduledFor: scheduledFor || new Date(),
    priority: type === 'event_cancelled' ? 'high' : 'medium'
  });
};

// Static method to create friend notification
notificationSchema.statics.createFriendNotification = function(userId, type, friendId, message) {
  return this.create({
    userId,
    type,
    title: 'Friend Activity',
    message,
    data: { friendId },
    priority: 'low'
  });
};

// Static method to create gamification notification
notificationSchema.statics.createGamificationNotification = function(userId, type, message, data = {}) {
  return this.create({
    userId,
    type,
    title: 'Achievement Unlocked!',
    message,
    data,
    priority: 'medium'
  });
};

// Static method to cleanup old notifications
notificationSchema.statics.cleanup = function(daysOld = 30) {
  const cutoffDate = new Date();
  cutoffDate.setDate(cutoffDate.getDate() - daysOld);
  
  return this.deleteMany({
    createdAt: { $lt: cutoffDate },
    status: { $in: ['read', 'failed'] }
  });
};

module.exports = mongoose.model('Notification', notificationSchema);
