const mongoose = require('mongoose');

const AgentMessageSchema = new mongoose.Schema({
  role: {
    type: String,
    enum: ['system', 'user', 'assistant'],
    required: true
  },
  content: {
    type: String,
    required: true
  },
  metadata: {
    tokensUsed: { type: Number, default: 0 },
    model: { type: String },
    reasoning: { type: String },
  },
  createdAt: {
    type: Date,
    default: Date.now
  }
}, { _id: false });

const AgentSessionSchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true,
    index: true
  },
  title: {
    type: String,
    default: 'New Agent Session'
  },
  messages: {
    type: [AgentMessageSchema],
    default: []
  },
  settings: {
    model: { type: String },
    temperature: { type: Number },
    topP: { type: Number }
  },
  archived: {
    type: Boolean,
    default: false,
    index: true
  },
  metadata: {
    createdAt: { type: Date, default: Date.now },
    updatedAt: { type: Date, default: Date.now },
    lastAssistantTokens: { type: Number, default: 0 },
    totalTokens: { type: Number, default: 0 }
  }
});

AgentSessionSchema.pre('save', function(next) {
  this.metadata.updatedAt = new Date();
  next();
});

module.exports = mongoose.model('AgentSession', AgentSessionSchema);


