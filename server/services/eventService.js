const Event = require('../models/Event');
const User = require('../models/User');
const externalApiService = require('./externalApiService');
const aiOrchestrator = require('./aiOrchestrator');
const cacheService = require('./cacheService');
const logger = require('../utils/logger');

class EventService {
  async createEvent(userId, eventData) {
    try {
      // Get user data
      const user = await User.findById(userId);
      if (!user) {
        throw new Error('User not found');
      }

      // Create event
      const event = new Event({
        ...eventData,
        host: {
          userId: userId,
          name: user.name,
          email: user.email,
          isUserHosted: true
        },
        status: 'published'
      });

      await event.save();

      // Get AI suggestions for the event
      try {
        const suggestions = await aiOrchestrator.generateEventSuggestions(event, user.preferences);
        event.aiSuggestions = suggestions;
        await event.save();
      } catch (error) {
        logger.warn('Failed to generate AI suggestions for event:', error);
      }

      logger.info(`Event created: ${event.name} by user ${userId}`);
      return event;

    } catch (error) {
      logger.error('Failed to create event:', error);
      throw new Error(`Failed to create event: ${error.message}`);
    }
  }

  async getEvent(eventId) {
    try {
      const event = await Event.findById(eventId)
        .populate('host.userId', 'name profilePicture')
        .populate('attendees.userId', 'name profilePicture')
        .populate('reviews.userId', 'name profilePicture');

      if (!event) {
        throw new Error('Event not found');
      }

      return event;
    } catch (error) {
      logger.error('Failed to get event:', error);
      throw new Error(`Failed to get event: ${error.message}`);
    }
  }

  async getNearbyEvents(location, radius = 5000, filters = {}) {
    try {
      const { category, startTime, endTime, limit = 20, offset = 0 } = filters;

      let events;
      if (category) {
        events = await Event.findByCategory(category, location.lat, location.lng, radius);
      } else {
        events = await Event.findByLocationAndTime(location.lat, location.lng, radius, startTime, endTime);
      }

      // Apply pagination
      const paginatedEvents = events.slice(offset, offset + limit);

      return {
        events: paginatedEvents,
        pagination: {
          total: events.length,
          limit,
          offset,
          hasMore: events.length > offset + limit
        }
      };

    } catch (error) {
      logger.error('Failed to get nearby events:', error);
      throw new Error(`Failed to get nearby events: ${error.message}`);
    }
  }

  async updateEvent(eventId, userId, updateData) {
    try {
      const event = await Event.findOne({
        _id: eventId,
        'host.userId': userId
      });

      if (!event) {
        throw new Error('Event not found or you are not the host');
      }

      // Update event
      Object.assign(event, updateData);
      await event.save();

      logger.info(`Event updated: ${event.name} by user ${userId}`);
      return event;

    } catch (error) {
      logger.error('Failed to update event:', error);
      throw new Error(`Failed to update event: ${error.message}`);
    }
  }

  async joinEvent(eventId, userId) {
    try {
      const event = await Event.findById(eventId);
      if (!event) {
        throw new Error('Event not found');
      }

      await event.addAttendee(userId);

      logger.info(`User ${userId} joined event ${eventId}`);
      return event;

    } catch (error) {
      logger.error('Failed to join event:', error);
      throw new Error(`Failed to join event: ${error.message}`);
    }
  }

  async leaveEvent(eventId, userId) {
    try {
      const event = await Event.findById(eventId);
      if (!event) {
        throw new Error('Event not found');
      }

      await event.removeAttendee(userId);

      logger.info(`User ${userId} left event ${eventId}`);
      return event;

    } catch (error) {
      logger.error('Failed to leave event:', error);
      throw new Error(`Failed to leave event: ${error.message}`);
    }
  }

  async reviewEvent(eventId, userId, rating, comment) {
    try {
      const event = await Event.findById(eventId);
      if (!event) {
        throw new Error('Event not found');
      }

      await event.addReview(userId, rating, comment);

      logger.info(`User ${userId} reviewed event ${eventId} with rating ${rating}`);
      return event;

    } catch (error) {
      logger.error('Failed to review event:', error);
      throw new Error(`Failed to review event: ${error.message}`);
    }
  }

  async cancelEvent(eventId, userId) {
    try {
      const event = await Event.findOne({
        _id: eventId,
        'host.userId': userId
      });

      if (!event) {
        throw new Error('Event not found or you are not the host');
      }

      await event.updateStatus('cancelled');

      logger.info(`Event ${eventId} cancelled by user ${userId}`);
      return event;

    } catch (error) {
      logger.error('Failed to cancel event:', error);
      throw new Error(`Failed to cancel event: ${error.message}`);
    }
  }

  async getUserEvents(userId, type = 'all', limit = 20, offset = 0) {
    try {
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
        .limit(limit)
        .skip(offset)
        .populate('host.userId', 'name profilePicture')
        .populate('attendees.userId', 'name profilePicture');

      return events;

    } catch (error) {
      logger.error('Failed to get user events:', error);
      throw new Error(`Failed to get user events: ${error.message}`);
    }
  }

  async getEventCategories() {
    try {
      const categories = [
        { id: 'nightlife', name: 'Nightlife', icon: '🌃', description: 'Bars, clubs, and evening entertainment' },
        { id: 'music', name: 'Music', icon: '🎵', description: 'Concerts, live music, and DJ sets' },
        { id: 'food', name: 'Food & Drink', icon: '🍽️', description: 'Restaurants, food trucks, and tastings' },
        { id: 'social', name: 'Social', icon: '👥', description: 'Meetups, networking, and social gatherings' },
        { id: 'culture', name: 'Culture', icon: '🎭', description: 'Museums, galleries, and cultural events' },
        { id: 'sports', name: 'Sports', icon: '⚽', description: 'Games, tournaments, and fitness events' },
        { id: 'outdoor', name: 'Outdoor', icon: '🌲', description: 'Hiking, parks, and outdoor activities' },
        { id: 'art', name: 'Art', icon: '🎨', description: 'Exhibitions, workshops, and creative events' },
        { id: 'education', name: 'Education', icon: '📚', description: 'Workshops, lectures, and learning events' }
      ];

      return categories;

    } catch (error) {
      logger.error('Failed to get event categories:', error);
      throw new Error(`Failed to get event categories: ${error.message}`);
    }
  }

  async getEventAnalytics(eventId, userId) {
    try {
      const event = await Event.findOne({
        _id: eventId,
        'host.userId': userId
      });

      if (!event) {
        throw new Error('Event not found or you are not the host');
      }

      const analytics = {
        totalViews: event.analytics.views,
        totalShares: event.analytics.shares,
        totalSaves: event.analytics.saves,
        attendeeCount: event.attendeeCount,
        capacity: event.capacity,
        attendanceRate: event.capacity > 0 ? (event.attendeeCount / event.capacity) * 100 : 0,
        averageRating: event.averageRating,
        totalReviews: event.reviews.length,
        waitlistCount: event.waitlist.length,
        completionRate: event.analytics.completionRate || 0
      };

      return analytics;

    } catch (error) {
      logger.error('Failed to get event analytics:', error);
      throw new Error(`Failed to get event analytics: ${error.message}`);
    }
  }

  async getPopularEvents(location, limit = 10) {
    try {
      const events = await Event.find({
        'location.lat': {
          $gte: location.lat - 0.01,
          $lte: location.lat + 0.01
        },
        'location.lng': {
          $gte: location.lng - 0.01,
          $lte: location.lng + 0.01
        },
        status: 'published',
        visibility: 'public',
        startTime: { $gte: new Date() }
      })
      .sort({ 'analytics.views': -1, 'reviews.rating': -1 })
      .limit(limit)
      .populate('host.userId', 'name profilePicture');

      return events;

    } catch (error) {
      logger.error('Failed to get popular events:', error);
      throw new Error(`Failed to get popular events: ${error.message}`);
    }
  }

  async getTrendingEvents(location, timeRange = '7d', limit = 10) {
    try {
      const timeRangeMs = this.getTimeRangeMs(timeRange);
      const startDate = new Date(Date.now() - timeRangeMs);

      const events = await Event.find({
        'location.lat': {
          $gte: location.lat - 0.01,
          $lte: location.lat + 0.01
        },
        'location.lng': {
          $gte: location.lng - 0.01,
          $lte: location.lng + 0.01
        },
        status: 'published',
        visibility: 'public',
        startTime: { $gte: new Date() },
        createdAt: { $gte: startDate }
      })
      .sort({ 'analytics.shares': -1, 'analytics.views': -1 })
      .limit(limit)
      .populate('host.userId', 'name profilePicture');

      return events;

    } catch (error) {
      logger.error('Failed to get trending events:', error);
      throw new Error(`Failed to get trending events: ${error.message}`);
    }
  }

  async searchEvents(query, location, radius = 5000, limit = 20) {
    try {
      const events = await Event.find({
        $or: [
          { name: { $regex: query, $options: 'i' } },
          { description: { $regex: query, $options: 'i' } },
          { tags: { $in: [new RegExp(query, 'i')] } }
        ],
        'location.lat': {
          $gte: location.lat - (radius / 111000),
          $lte: location.lat + (radius / 111000)
        },
        'location.lng': {
          $gte: location.lng - (radius / (111000 * Math.cos(location.lat * Math.PI / 180))),
          $lte: location.lng + (radius / (111000 * Math.cos(location.lat * Math.PI / 180)))
        },
        status: 'published',
        visibility: 'public',
        startTime: { $gte: new Date() }
      })
      .sort({ startTime: 1 })
      .limit(limit)
      .populate('host.userId', 'name profilePicture');

      return events;

    } catch (error) {
      logger.error('Failed to search events:', error);
      throw new Error(`Failed to search events: ${error.message}`);
    }
  }

  async syncExternalEvents(location, radius = 5000) {
    try {
      // Get events from external APIs
      const externalEvents = await externalApiService.getNearbyEvents(location, radius);

      let syncedCount = 0;
      for (const externalEvent of externalEvents) {
        try {
          // Check if event already exists
          const existingEvent = await Event.findOne({
            'metadata.externalId': externalEvent.eventId,
            'metadata.source': 'eventbrite'
          });

          if (!existingEvent) {
            // Create new event from external data
            const event = new Event({
              name: externalEvent.name,
              description: externalEvent.description,
              type: this.mapExternalEventType(externalEvent.category),
              category: this.mapExternalEventCategory(externalEvent.category),
              location: externalEvent.location,
              startTime: externalEvent.startTime,
              endTime: externalEvent.endTime,
              duration: Math.round((externalEvent.endTime - externalEvent.startTime) / (1000 * 60)),
              capacity: externalEvent.venue.capacity,
              price: externalEvent.price,
              status: 'published',
              visibility: 'public',
              metadata: {
                source: 'eventbrite',
                externalId: externalEvent.eventId,
                lastSynced: new Date()
              }
            });

            await event.save();
            syncedCount++;
          } else {
            // Update existing event
            existingEvent.metadata.lastSynced = new Date();
            await existingEvent.save();
          }
        } catch (error) {
          logger.warn(`Failed to sync external event ${externalEvent.eventId}:`, error);
        }
      }

      logger.info(`Synced ${syncedCount} external events`);
      return syncedCount;

    } catch (error) {
      logger.error('Failed to sync external events:', error);
      throw new Error(`Failed to sync external events: ${error.message}`);
    }
  }

  mapExternalEventType(category) {
    const typeMapping = {
      'music': 'concert',
      'food': 'restaurant',
      'nightlife': 'bar',
      'social': 'meetup',
      'culture': 'cultural',
      'sports': 'sports',
      'outdoor': 'outdoor',
      'art': 'art',
      'education': 'workshop'
    };
    return typeMapping[category] || 'meetup';
  }

  mapExternalEventCategory(category) {
    const categoryMapping = {
      'music': 'music',
      'food': 'food',
      'nightlife': 'nightlife',
      'social': 'social',
      'culture': 'culture',
      'sports': 'sports',
      'outdoor': 'outdoor',
      'art': 'art',
      'education': 'education'
    };
    return categoryMapping[category] || 'social';
  }

  getTimeRangeMs(timeRange) {
    const ranges = {
      '1d': 24 * 60 * 60 * 1000,
      '7d': 7 * 24 * 60 * 60 * 1000,
      '30d': 30 * 24 * 60 * 60 * 1000,
      '90d': 90 * 24 * 60 * 60 * 1000
    };
    return ranges[timeRange] || ranges['7d'];
  }
}

module.exports = new EventService();
