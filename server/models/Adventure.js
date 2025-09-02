const mongoose = require('mongoose');

const locationSchema = new mongoose.Schema({
  lat: { type: Number, required: true },
  lng: { type: Number, required: true },
  address: { type: String, required: true },
  placeId: String
});

const adventureStepSchema = new mongoose.Schema({
  type: {
    type: String,
    enum: ['event', 'venue', 'travel', 'activity', 'break'],
    required: true
  },
  name: { type: String, required: true },
  description: String,
  location: locationSchema,
  startTime: { type: Date, required: true },
  endTime: { type: Date, required: true },
  duration: { type: Number, required: true }, // in minutes
  travelMethod: {
    type: String,
    enum: ['walk', 'bike', 'drive', 'transit', 'taxi'],
    default: 'walk'
  },
  travelDuration: Number, // in minutes
  travelDistance: Number, // in meters
  task: {
    description: String,
    points: { type: Number, default: 0 },
    completed: { type: Boolean, default: false }
  },
  venue: {
    venueId: String,
    type: String,
    rating: Number,
    priceLevel: Number,
    photos: [String],
    website: String,
    phone: String
  },
  event: {
    eventId: String,
    type: String,
    capacity: Number,
    attendees: Number,
    hostId: String,
    isUserHosted: { type: Boolean, default: false }
  },
  weather: {
    condition: String,
    temperature: Number,
    humidity: Number,
    windSpeed: Number
  },
  alternatives: [{
    name: String,
    location: locationSchema,
    reason: String
  }]
});

const adventureSchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  name: {
    type: String,
    required: true,
    maxlength: 100
  },
  description: {
    type: String,
    maxlength: 500
  },
  status: {
    type: String,
    enum: ['generated', 'active', 'completed', 'cancelled', 'paused'],
    default: 'generated'
  },
  steps: [adventureStepSchema],
  totalDuration: {
    type: Number,
    required: true,
    min: 30,
    max: 90
  },
  totalDistance: Number, // in meters
  startLocation: locationSchema,
  endLocation: locationSchema,
  preferences: {
    interests: [String],
    skillLevel: String,
    socialMode: String,
    budget: String,
    maxDistance: Number
  },
  social: {
    friendsInvited: [{
      type: mongoose.Schema.Types.ObjectId,
      ref: 'User'
    }],
    friendsJoined: [{
      type: mongoose.Schema.Types.ObjectId,
      ref: 'User'
    }],
    isPublic: { type: Boolean, default: false },
    maxParticipants: Number
  },
  weather: {
    forecast: [{
      time: Date,
      condition: String,
      temperature: Number,
      humidity: Number,
      windSpeed: Number
    }],
    alerts: [String]
  },
  aiMetadata: {
    model: String,
    version: String,
    generationTime: Number,
    tokensUsed: Number,
    confidence: Number,
    reasoning: String
  },
  feedback: {
    rating: {
      type: Number,
      min: 1,
      max: 5
    },
    comments: String,
    completedSteps: [String],
    skippedSteps: [String],
    suggestions: String,
    submittedAt: Date
  },
  gamification: {
    points: { type: Number, default: 0 },
    badges: [String],
    challenges: [{
      description: String,
      completed: { type: Boolean, default: false },
      points: Number
    }]
  },
  notifications: [{
    type: {
      type: String,
      enum: ['step_reminder', 'weather_alert', 'venue_change', 'friend_joined', 'completion']
    },
    message: String,
    sentAt: Date,
    read: { type: Boolean, default: false }
  }],
  metadata: {
    generatedAt: { type: Date, default: Date.now },
    startedAt: Date,
    completedAt: Date,
    lastUpdated: { type: Date, default: Date.now },
    version: { type: String, default: '2.0' }
  }
}, {
  timestamps: true
});

// Indexes for better query performance
adventureSchema.index({ userId: 1, status: 1 });
adventureSchema.index({ 'startLocation.lat': 1, 'startLocation.lng': 1 });
adventureSchema.index({ 'metadata.generatedAt': -1 });
adventureSchema.index({ 'social.friendsInvited': 1 });

// Virtual for total points
adventureSchema.virtual('totalPoints').get(function() {
  return this.gamification.points + 
    this.steps.reduce((total, step) => total + (step.task?.points || 0), 0);
});

// Method to update adventure status
adventureSchema.methods.updateStatus = function(newStatus) {
  this.status = newStatus;
  this.metadata.lastUpdated = new Date();
  
  if (newStatus === 'active' && !this.metadata.startedAt) {
    this.metadata.startedAt = new Date();
  } else if (newStatus === 'completed' && !this.metadata.completedAt) {
    this.metadata.completedAt = new Date();
  }
  
  return this.save();
};

// Method to add notification
adventureSchema.methods.addNotification = function(type, message) {
  this.notifications.push({
    type,
    message,
    sentAt: new Date(),
    read: false
  });
  return this.save();
};

// Method to mark step as completed
adventureSchema.methods.completeStep = function(stepIndex) {
  if (this.steps[stepIndex]) {
    this.steps[stepIndex].task.completed = true;
    this.metadata.lastUpdated = new Date();
    return this.save();
  }
  throw new Error('Step not found');
};

// Method to get current step
adventureSchema.methods.getCurrentStep = function() {
  const now = new Date();
  return this.steps.find(step => 
    step.startTime <= now && step.endTime >= now
  );
};

// Method to get next step
adventureSchema.methods.getNextStep = function() {
  const now = new Date();
  return this.steps.find(step => step.startTime > now);
};

// Static method to find adventures by location
adventureSchema.statics.findByLocation = function(lat, lng, radius = 5000) {
  return this.find({
    'startLocation.lat': {
      $gte: lat - (radius / 111000), // Rough conversion from meters to degrees
      $lte: lat + (radius / 111000)
    },
    'startLocation.lng': {
      $gte: lng - (radius / (111000 * Math.cos(lat * Math.PI / 180))),
      $lte: lng + (radius / (111000 * Math.cos(lat * Math.PI / 180)))
    },
    status: { $in: ['generated', 'active'] }
  });
};

module.exports = mongoose.model('Adventure', adventureSchema);
