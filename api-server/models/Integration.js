const mongoose = require('mongoose');

const integrationSchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true,
    index: true
  },
  service: {
    type: String,
    enum: ['notion', 'trello', 'github', 'google_docs', 'slack', 'todoist', 'asana'],
    required: true
  },
  status: {
    type: String,
    enum: ['connected', 'disconnected', 'error', 'syncing'],
    default: 'connected'
  },
  credentials: {
    // Encrypted tokens/keys
    accessToken: String,
    refreshToken: String,
    tokenExpiresAt: Date,
    apiKey: String
  },
  configuration: {
    autoSync: {
      type: Boolean,
      default: true
    },
    syncInterval: {
      type: Number,
      default: 3600 // seconds
    },
    syncFilters: {
      // Service-specific filters
      labels: [String],
      projects: [String],
      statuses: [String]
    }
  },
  lastSync: {
    type: Date
  },
  syncStats: {
    totalTasksSynced: {
      type: Number,
      default: 0
    },
    lastSyncSuccess: {
      type: Boolean,
      default: true
    },
    lastSyncError: String
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

integrationSchema.index({ userId: 1, service: 1 });
integrationSchema.index({ userId: 1, status: 1 });
integrationSchema.index({ 'credentials.tokenExpiresAt': 1 });

integrationSchema.pre('save', function(next) {
  this.updatedAt = Date.now();
  next();
});

module.exports = mongoose.model('Integration', integrationSchema);

