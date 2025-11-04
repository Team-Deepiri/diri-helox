const mongoose = require('mongoose');

const locationSchema = new mongoose.Schema({
  lat: { type: Number, required: true },
  lng: { type: Number, required: true },
  address: { type: String, required: true },
  placeId: String,
  venue: {
    name: String,
    type: String,
    rating: Number,
    priceLevel: Number,
    photos: [String],
    website: String,
    phone: String
  }
});

const eventSchema = new mongoose.Schema({
  name: {
    type: String,
    required: true,
    maxlength: 150
  },
  description: {
    type: String,
    maxlength: 1000
  },
  type: {
    type: String,
    enum: ['bar', 'restaurant', 'concert', 'popup', 'meetup', 'party', 'workshop', 'sports', 'cultural', 'outdoor'],
    required: true
  },
  category: {
    type: String,
    enum: ['nightlife', 'music', 'food', 'social', 'culture', 'sports', 'outdoor', 'art', 'education'],
    required: true
  },
  location: locationSchema,
  startTime: {
    type: Date,
    required: true
  },
  endTime: {
    type: Date,
    required: true
  },
  duration: {
    type: Number,
    required: true // in minutes
  },
  host: {
    userId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'User'
    },
    name: String,
    email: String,
    isUserHosted: { type: Boolean, default: false }
  },
  capacity: {
    type: Number,
    min: 1,
    max: 1000
  },
  attendees: [{
    userId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'User'
    },
    joinedAt: { type: Date, default: Date.now },
    status: {
      type: String,
      enum: ['confirmed', 'waitlist', 'cancelled'],
      default: 'confirmed'
    }
  }],
  waitlist: [{
    userId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'User'
    },
    joinedAt: { type: Date, default: Date.now }
  }],
  price: {
    amount: { type: Number, default: 0 },
    currency: { type: String, default: 'USD' },
    isFree: { type: Boolean, default: true }
  },
  requirements: {
    ageRestriction: {
      min: Number,
      max: Number
    },
    skillLevel: {
      type: String,
      enum: ['beginner', 'intermediate', 'advanced', 'any']
    },
    equipment: [String],
    dressCode: String
  },
  tags: [String],
  images: [String],
  externalLinks: {
    website: String,
    facebook: String,
    instagram: String,
    ticketUrl: String
  },
  status: {
    type: String,
    enum: ['draft', 'published', 'cancelled', 'completed', 'postponed'],
    default: 'draft'
  },
  visibility: {
    type: String,
    enum: ['public', 'friends', 'private'],
    default: 'public'
  },
  weather: {
    condition: String,
    temperature: Number,
    humidity: Number,
    windSpeed: Number,
    lastUpdated: Date
  },
  aiSuggestions: {
    bestTimeSlots: [Date],
    nearbyAttractions: [String],
    similarEvents: [String],
    recommendations: String
  },
  analytics: {
    views: { type: Number, default: 0 },
    shares: { type: Number, default: 0 },
    saves: { type: Number, default: 0 },
    completionRate: Number
  },
  reviews: [{
    userId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'User'
    },
    rating: {
      type: Number,
      min: 1,
      max: 5
    },
    comment: String,
    createdAt: { type: Date, default: Date.now }
  }],
  metadata: {
    source: {
      type: String,
      enum: ['user', 'eventbrite', 'yelp', 'google_places', 'manual'],
      default: 'user'
    },
    externalId: String,
    lastSynced: Date,
    version: { type: String, default: '2.0' }
  }
}, {
  timestamps: true
});

// Indexes for better query performance
eventSchema.index({ 'location.lat': 1, 'location.lng': 1 });
eventSchema.index({ startTime: 1, endTime: 1 });
eventSchema.index({ type: 1, category: 1 });
eventSchema.index({ status: 1, visibility: 1 });
eventSchema.index({ 'host.userId': 1 });
eventSchema.index({ tags: 1 });

// Virtual for current attendees count
eventSchema.virtual('attendeeCount').get(function() {
  return this.attendees.filter(attendee => attendee.status === 'confirmed').length;
});

// Virtual for available spots
eventSchema.virtual('availableSpots').get(function() {
  return Math.max(0, this.capacity - this.attendeeCount);
});

// Virtual for average rating
eventSchema.virtual('averageRating').get(function() {
  if (this.reviews.length === 0) return 0;
  const sum = this.reviews.reduce((total, review) => total + review.rating, 0);
  return sum / this.reviews.length;
});

// Method to add attendee
eventSchema.methods.addAttendee = function(userId) {
  // Check if user is already attending
  const existingAttendee = this.attendees.find(attendee => 
    attendee.userId.toString() === userId.toString()
  );
  
  if (existingAttendee) {
    throw new Error('User is already attending this event');
  }
  
  // Check capacity
  if (this.attendeeCount >= this.capacity) {
    // Add to waitlist
    this.waitlist.push({ userId, joinedAt: new Date() });
    return this.save();
  }
  
  // Add as confirmed attendee
  this.attendees.push({
    userId,
    joinedAt: new Date(),
    status: 'confirmed'
  });
  
  return this.save();
};

// Method to remove attendee
eventSchema.methods.removeAttendee = function(userId) {
  // Remove from attendees
  this.attendees = this.attendees.filter(attendee => 
    attendee.userId.toString() !== userId.toString()
  );
  
  // Remove from waitlist
  this.waitlist = this.waitlist.filter(attendee => 
    attendee.userId.toString() !== userId.toString()
  );
  
  // Move first person from waitlist to attendees if there's space
  if (this.waitlist.length > 0 && this.attendeeCount < this.capacity) {
    const nextInLine = this.waitlist.shift();
    this.attendees.push({
      userId: nextInLine.userId,
      joinedAt: new Date(),
      status: 'confirmed'
    });
  }
  
  return this.save();
};

// Method to update event status
eventSchema.methods.updateStatus = function(newStatus) {
  this.status = newStatus;
  return this.save();
};

// Method to add review
eventSchema.methods.addReview = function(userId, rating, comment) {
  // Check if user attended the event
  const attended = this.attendees.some(attendee => 
    attendee.userId.toString() === userId.toString() && 
    attendee.status === 'confirmed'
  );
  
  if (!attended) {
    throw new Error('Only attendees can review events');
  }
  
  // Check if user already reviewed
  const existingReview = this.reviews.find(review => 
    review.userId.toString() === userId.toString()
  );
  
  if (existingReview) {
    throw new Error('User has already reviewed this event');
  }
  
  this.reviews.push({
    userId,
    rating,
    comment,
    createdAt: new Date()
  });
  
  return this.save();
};

// Static method to find events by location and time
eventSchema.statics.findByLocationAndTime = function(lat, lng, radius = 5000, startTime, endTime) {
  const query = {
    'location.lat': {
      $gte: lat - (radius / 111000),
      $lte: lat + (radius / 111000)
    },
    'location.lng': {
      $gte: lng - (radius / (111000 * Math.cos(lat * Math.PI / 180))),
      $lte: lng + (radius / (111000 * Math.cos(lat * Math.PI / 180)))
    },
    status: 'published',
    visibility: 'public'
  };
  
  if (startTime && endTime) {
    query.startTime = { $gte: startTime, $lte: endTime };
  }
  
  return this.find(query).sort({ startTime: 1 });
};

// Static method to find events by category
eventSchema.statics.findByCategory = function(category, lat, lng, radius = 5000) {
  return this.findByLocationAndTime(lat, lng, radius).where('category', category);
};

module.exports = mongoose.model('Event', eventSchema);
