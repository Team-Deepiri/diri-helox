const express = require('express');
const Joi = require('joi');
const Event = require('../models/Event');
const logger = require('../utils/logger');

const router = express.Router();

// Validation schemas
const createEventSchema = Joi.object({
  name: Joi.string().min(2).max(150).required(),
  description: Joi.string().max(1000).optional(),
  type: Joi.string().valid('bar', 'restaurant', 'concert', 'popup', 'meetup', 'party', 'workshop', 'sports', 'cultural', 'outdoor').required(),
  category: Joi.string().valid('nightlife', 'music', 'food', 'social', 'culture', 'sports', 'outdoor', 'art', 'education').required(),
  location: Joi.object({
    lat: Joi.number().min(-90).max(90).required(),
    lng: Joi.number().min(-180).max(180).required(),
    address: Joi.string().required()
  }).required(),
  startTime: Joi.date().required(),
  endTime: Joi.date().required(),
  capacity: Joi.number().min(1).max(1000).optional(),
  price: Joi.object({
    amount: Joi.number().min(0).optional(),
    currency: Joi.string().default('USD'),
    isFree: Joi.boolean().default(true)
  }).optional(),
  requirements: Joi.object({
    ageRestriction: Joi.object({
      min: Joi.number().min(0).optional(),
      max: Joi.number().max(120).optional()
    }).optional(),
    skillLevel: Joi.string().valid('beginner', 'intermediate', 'advanced', 'any').optional(),
    equipment: Joi.array().items(Joi.string()).optional(),
    dressCode: Joi.string().optional()
  }).optional(),
  tags: Joi.array().items(Joi.string()).optional(),
  visibility: Joi.string().valid('public', 'friends', 'private').default('public')
});

const updateEventSchema = Joi.object({
  name: Joi.string().min(2).max(150).optional(),
  description: Joi.string().max(1000).optional(),
  startTime: Joi.date().optional(),
  endTime: Joi.date().optional(),
  capacity: Joi.number().min(1).max(1000).optional(),
  price: Joi.object({
    amount: Joi.number().min(0).optional(),
    currency: Joi.string().optional(),
    isFree: Joi.boolean().optional()
  }).optional(),
  requirements: Joi.object({
    ageRestriction: Joi.object({
      min: Joi.number().min(0).optional(),
      max: Joi.number().max(120).optional()
    }).optional(),
    skillLevel: Joi.string().valid('beginner', 'intermediate', 'advanced', 'any').optional(),
    equipment: Joi.array().items(Joi.string()).optional(),
    dressCode: Joi.string().optional()
  }).optional(),
  tags: Joi.array().items(Joi.string()).optional(),
  visibility: Joi.string().valid('public', 'friends', 'private').optional()
});

const reviewEventSchema = Joi.object({
  rating: Joi.number().min(1).max(5).required(),
  comment: Joi.string().max(500).optional()
});

// Create new event
router.post('/', async (req, res) => {
  try {
    const userId = req.user.userId;
    
    // Validate request data
    const { error, value } = createEventSchema.validate(req.body);
    if (error) {
      return res.status(400).json({
        success: false,
        message: 'Validation error',
        errors: error.details.map(detail => detail.message)
      });
    }

    // Create event
    const event = new Event({
      ...value,
      host: {
        userId: userId,
        name: req.user.name || 'Anonymous',
        email: req.user.email,
        isUserHosted: true
      },
      status: 'published'
    });

    await event.save();

    // Send real-time notification
    if (req.io) {
      req.io.emit('event_created', {
        eventId: event._id,
        name: event.name,
        message: 'New event created!'
      });
    }

    res.status(201).json({
      success: true,
      message: 'Event created successfully',
      data: event
    });

  } catch (error) {
    logger.error('Event creation failed:', error);
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
});

// Get events near location
router.get('/nearby', async (req, res) => {
  try {
    const { lat, lng, radius = 5000, category, limit = 20, offset = 0 } = req.query;

    if (!lat || !lng) {
      return res.status(400).json({
        success: false,
        message: 'Latitude and longitude are required'
      });
    }

    const location = { lat: parseFloat(lat), lng: parseFloat(lng) };
    const startTime = req.query.startTime ? new Date(req.query.startTime) : new Date();
    const endTime = req.query.endTime ? new Date(req.query.endTime) : new Date(Date.now() + 7 * 24 * 60 * 60 * 1000); // 7 days from now

    let events;
    if (category) {
      events = await Event.findByCategory(category, location.lat, location.lng, parseInt(radius));
    } else {
      events = await Event.findByLocationAndTime(location.lat, location.lng, parseInt(radius), startTime, endTime);
    }

    // Apply pagination
    const paginatedEvents = events.slice(parseInt(offset), parseInt(offset) + parseInt(limit));

    res.json({
      success: true,
      data: paginatedEvents,
      pagination: {
        total: events.length,
        limit: parseInt(limit),
        offset: parseInt(offset),
        hasMore: events.length > parseInt(offset) + parseInt(limit)
      }
    });

  } catch (error) {
    logger.error('Failed to get nearby events:', error);
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
});

// Get specific event
router.get('/:eventId', async (req, res) => {
  try {
    const { eventId } = req.params;
    const event = await Event.findById(eventId)
      .populate('host.userId', 'name profilePicture')
      .populate('attendees.userId', 'name profilePicture')
      .populate('reviews.userId', 'name profilePicture');

    if (!event) {
      return res.status(404).json({
        success: false,
        message: 'Event not found'
      });
    }

    res.json({
      success: true,
      data: event
    });

  } catch (error) {
    logger.error('Failed to get event:', error);
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
});

// Update event
router.put('/:eventId', async (req, res) => {
  try {
    const userId = req.user.userId;
    const { eventId } = req.params;

    // Validate request data
    const { error, value } = updateEventSchema.validate(req.body);
    if (error) {
      return res.status(400).json({
        success: false,
        message: 'Validation error',
        errors: error.details.map(detail => detail.message)
      });
    }

    const event = await Event.findOne({
      _id: eventId,
      'host.userId': userId
    });

    if (!event) {
      return res.status(404).json({
        success: false,
        message: 'Event not found or you are not the host'
      });
    }

    // Update event
    Object.assign(event, value);
    await event.save();

    // Send real-time notification
    if (req.io) {
      req.io.to(`event_${eventId}`).emit('event_updated', {
        eventId: event._id,
        name: event.name,
        message: 'Event has been updated'
      });
    }

    res.json({
      success: true,
      message: 'Event updated successfully',
      data: event
    });

  } catch (error) {
    logger.error('Failed to update event:', error);
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
});

// Join event
router.post('/:eventId/join', async (req, res) => {
  try {
    const userId = req.user.userId;
    const { eventId } = req.params;

    const event = await Event.findById(eventId);
    if (!event) {
      return res.status(404).json({
        success: false,
        message: 'Event not found'
      });
    }

    await event.addAttendee(userId);

    // Send real-time notification
    if (req.io) {
      req.io.to(`event_${eventId}`).emit('user_joined', {
        userId: userId,
        eventId: eventId,
        message: 'Someone joined the event'
      });
    }

    res.json({
      success: true,
      message: 'Successfully joined event',
      data: event
    });

  } catch (error) {
    logger.error('Failed to join event:', error);
    res.status(400).json({
      success: false,
      message: error.message
    });
  }
});

// Leave event
router.post('/:eventId/leave', async (req, res) => {
  try {
    const userId = req.user.userId;
    const { eventId } = req.params;

    const event = await Event.findById(eventId);
    if (!event) {
      return res.status(404).json({
        success: false,
        message: 'Event not found'
      });
    }

    await event.removeAttendee(userId);

    // Send real-time notification
    if (req.io) {
      req.io.to(`event_${eventId}`).emit('user_left', {
        userId: userId,
        eventId: eventId,
        message: 'Someone left the event'
      });
    }

    res.json({
      success: true,
      message: 'Successfully left event',
      data: event
    });

  } catch (error) {
    logger.error('Failed to leave event:', error);
    res.status(400).json({
      success: false,
      message: error.message
    });
  }
});

// Review event
router.post('/:eventId/review', async (req, res) => {
  try {
    const userId = req.user.userId;
    const { eventId } = req.params;

    // Validate request data
    const { error, value } = reviewEventSchema.validate(req.body);
    if (error) {
      return res.status(400).json({
        success: false,
        message: 'Validation error',
        errors: error.details.map(detail => detail.message)
      });
    }

    const event = await Event.findById(eventId);
    if (!event) {
      return res.status(404).json({
        success: false,
        message: 'Event not found'
      });
    }

    await event.addReview(userId, value.rating, value.comment);

    res.json({
      success: true,
      message: 'Review added successfully',
      data: event
    });

  } catch (error) {
    logger.error('Failed to add review:', error);
    res.status(400).json({
      success: false,
      message: error.message
    });
  }
});

// Cancel event
router.post('/:eventId/cancel', async (req, res) => {
  try {
    const userId = req.user.userId;
    const { eventId } = req.params;

    const event = await Event.findOne({
      _id: eventId,
      'host.userId': userId
    });

    if (!event) {
      return res.status(404).json({
        success: false,
        message: 'Event not found or you are not the host'
      });
    }

    await event.updateStatus('cancelled');

    // Send real-time notification to all attendees
    if (req.io) {
      req.io.to(`event_${eventId}`).emit('event_cancelled', {
        eventId: event._id,
        name: event.name,
        message: 'Event has been cancelled'
      });
    }

    res.json({
      success: true,
      message: 'Event cancelled successfully',
      data: event
    });

  } catch (error) {
    logger.error('Failed to cancel event:', error);
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
});

// Get user's events
router.get('/user/events', async (req, res) => {
  try {
    const userId = req.user.userId;
    const { type = 'all', limit = 20, offset = 0 } = req.query;

    let query = {};
    if (type === 'hosted') {
      query = { 'host.userId': userId };
    } else if (type === 'attending') {
      query = { 'attendees.userId': userId };
    } else {
      query = {
        $or: [
          { 'host.userId': userId },
          { 'attendees.userId': userId }
        ]
      };
    }

    const events = await Event.find(query)
      .sort({ startTime: -1 })
      .limit(parseInt(limit))
      .skip(parseInt(offset))
      .populate('host.userId', 'name profilePicture')
      .populate('attendees.userId', 'name profilePicture');

    res.json({
      success: true,
      data: events
    });

  } catch (error) {
    logger.error('Failed to get user events:', error);
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
});

// Get event categories
router.get('/categories', async (req, res) => {
  try {
    const categories = [
      { id: 'nightlife', name: 'Nightlife', icon: '🌃' },
      { id: 'music', name: 'Music', icon: '🎵' },
      { id: 'food', name: 'Food & Drink', icon: '🍽️' },
      { id: 'social', name: 'Social', icon: '👥' },
      { id: 'culture', name: 'Culture', icon: '🎭' },
      { id: 'sports', name: 'Sports', icon: '⚽' },
      { id: 'outdoor', name: 'Outdoor', icon: '🌲' },
      { id: 'art', name: 'Art', icon: '🎨' },
      { id: 'education', name: 'Education', icon: '📚' }
    ];

    res.json({
      success: true,
      data: categories
    });

  } catch (error) {
    logger.error('Failed to get categories:', error);
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
});

module.exports = router;
