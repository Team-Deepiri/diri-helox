const express = require('express');
const Joi = require('joi');
const userService = require('../services/userService');
const logger = require('../utils/logger');

const router = express.Router();

// Validation schemas
const updateUserSchema = Joi.object({
  name: Joi.string().min(2).max(100).optional(),
  profilePicture: Joi.string().uri().optional(),
  bio: Joi.string().max(500).optional(),
  location: Joi.object({
    lat: Joi.number().min(-90).max(90).required(),
    lng: Joi.number().min(-180).max(180).required(),
    address: Joi.string().required()
  }).optional()
});

const updatePreferencesSchema = Joi.object({
  interests: Joi.array().items(Joi.string()).optional(),
  skillLevel: Joi.string().valid('beginner', 'intermediate', 'advanced').optional(),
  maxDistance: Joi.number().min(1000).max(20000).optional(),
  preferredDuration: Joi.number().min(30).max(90).optional(),
  socialMode: Joi.string().valid('solo', 'friends', 'meet_new_people').optional(),
  budget: Joi.string().valid('low', 'medium', 'high').optional(),
  timePreferences: Joi.object({
    morning: Joi.boolean().optional(),
    afternoon: Joi.boolean().optional(),
    evening: Joi.boolean().optional(),
    night: Joi.boolean().optional()
  }).optional()
});

const addFriendSchema = Joi.object({
  friendId: Joi.string().required()
});

const searchUsersSchema = Joi.object({
  query: Joi.string().min(2).required(),
  limit: Joi.number().min(1).max(50).optional()
});

const addFavoriteVenueSchema = Joi.object({
  venueId: Joi.string().required(),
  name: Joi.string().required(),
  type: Joi.string().required(),
  location: Joi.object({
    lat: Joi.number().required(),
    lng: Joi.number().required(),
    address: Joi.string().required()
  }).required()
});

// Get current user profile
router.get('/profile', async (req, res) => {
  try {
    const userId = req.user.userId;
    const user = await userService.getUserById(userId);

    res.json({
      success: true,
      data: user.getPublicProfile()
    });

  } catch (error) {
    logger.error('Failed to get user profile:', error);
    res.status(404).json({
      success: false,
      message: error.message
    });
  }
});

// Update user profile
router.put('/profile', async (req, res) => {
  try {
    const userId = req.user.userId;
    
    // Validate request data
    const { error, value } = updateUserSchema.validate(req.body);
    if (error) {
      return res.status(400).json({
        success: false,
        message: 'Validation error',
        errors: error.details.map(detail => detail.message)
      });
    }

    const user = await userService.updateUser(userId, value);

    res.json({
      success: true,
      message: 'Profile updated successfully',
      data: user.getPublicProfile()
    });

  } catch (error) {
    logger.error('Failed to update user profile:', error);
    res.status(400).json({
      success: false,
      message: error.message
    });
  }
});

// Update user preferences
router.put('/preferences', async (req, res) => {
  try {
    const userId = req.user.userId;
    
    // Validate request data
    const { error, value } = updatePreferencesSchema.validate(req.body);
    if (error) {
      return res.status(400).json({
        success: false,
        message: 'Validation error',
        errors: error.details.map(detail => detail.message)
      });
    }

    const user = await userService.updateUserPreferences(userId, value);

    res.json({
      success: true,
      message: 'Preferences updated successfully',
      data: user.preferences
    });

  } catch (error) {
    logger.error('Failed to update user preferences:', error);
    res.status(400).json({
      success: false,
      message: error.message
    });
  }
});

// Update user location
router.put('/location', async (req, res) => {
  try {
    const userId = req.user.userId;
    const { lat, lng, address } = req.body;

    if (!lat || !lng || !address) {
      return res.status(400).json({
        success: false,
        message: 'Latitude, longitude, and address are required'
      });
    }

    const location = { lat: parseFloat(lat), lng: parseFloat(lng), address };
    const user = await userService.updateUserLocation(userId, location);

    res.json({
      success: true,
      message: 'Location updated successfully',
      data: user.location
    });

  } catch (error) {
    logger.error('Failed to update user location:', error);
    res.status(400).json({
      success: false,
      message: error.message
    });
  }
});

// Get user stats
router.get('/stats', async (req, res) => {
  try {
    const userId = req.user.userId;
    const stats = await userService.getUserStats(userId);

    res.json({
      success: true,
      data: stats
    });

  } catch (error) {
    logger.error('Failed to get user stats:', error);
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
});

// Get user's friends
router.get('/friends', async (req, res) => {
  try {
    const userId = req.user.userId;
    const friends = await userService.getFriends(userId);

    res.json({
      success: true,
      data: friends
    });

  } catch (error) {
    logger.error('Failed to get friends:', error);
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
});

// Add friend
router.post('/friends', async (req, res) => {
  try {
    const userId = req.user.userId;
    
    // Validate request data
    const { error, value } = addFriendSchema.validate(req.body);
    if (error) {
      return res.status(400).json({
        success: false,
        message: 'Validation error',
        errors: error.details.map(detail => detail.message)
      });
    }

    const result = await userService.addFriend(userId, value.friendId);

    res.json({
      success: true,
      message: 'Friend added successfully',
      data: {
        friend: result.friend.getPublicProfile()
      }
    });

  } catch (error) {
    logger.error('Failed to add friend:', error);
    res.status(400).json({
      success: false,
      message: error.message
    });
  }
});

// Remove friend
router.delete('/friends/:friendId', async (req, res) => {
  try {
    const userId = req.user.userId;
    const { friendId } = req.params;

    const result = await userService.removeFriend(userId, friendId);

    res.json({
      success: true,
      message: 'Friend removed successfully',
      data: {
        friend: result.friend.getPublicProfile()
      }
    });

  } catch (error) {
    logger.error('Failed to remove friend:', error);
    res.status(400).json({
      success: false,
      message: error.message
    });
  }
});

// Search users
router.get('/search', async (req, res) => {
  try {
    const userId = req.user.userId;
    
    // Validate request data
    const { error, value } = searchUsersSchema.validate(req.query);
    if (error) {
      return res.status(400).json({
        success: false,
        message: 'Validation error',
        errors: error.details.map(detail => detail.message)
      });
    }

    const users = await userService.searchUsers(value.query, userId, value.limit);

    res.json({
      success: true,
      data: users
    });

  } catch (error) {
    logger.error('Failed to search users:', error);
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
});

// Get user's favorite venues
router.get('/favorite-venues', async (req, res) => {
  try {
    const userId = req.user.userId;
    const user = await userService.getUserById(userId);

    res.json({
      success: true,
      data: user.favoriteVenues
    });

  } catch (error) {
    logger.error('Failed to get favorite venues:', error);
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
});

// Add favorite venue
router.post('/favorite-venues', async (req, res) => {
  try {
    const userId = req.user.userId;
    
    // Validate request data
    const { error, value } = addFavoriteVenueSchema.validate(req.body);
    if (error) {
      return res.status(400).json({
        success: false,
        message: 'Validation error',
        errors: error.details.map(detail => detail.message)
      });
    }

    const user = await userService.addFavoriteVenue(userId, value);

    res.json({
      success: true,
      message: 'Venue added to favorites',
      data: user.favoriteVenues
    });

  } catch (error) {
    logger.error('Failed to add favorite venue:', error);
    res.status(400).json({
      success: false,
      message: error.message
    });
  }
});

// Remove favorite venue
router.delete('/favorite-venues/:venueId', async (req, res) => {
  try {
    const userId = req.user.userId;
    const { venueId } = req.params;

    const user = await userService.removeFavoriteVenue(userId, venueId);

    res.json({
      success: true,
      message: 'Venue removed from favorites',
      data: user.favoriteVenues
    });

  } catch (error) {
    logger.error('Failed to remove favorite venue:', error);
    res.status(400).json({
      success: false,
      message: error.message
    });
  }
});

// Get leaderboard
router.get('/leaderboard', async (req, res) => {
  try {
    const { timeRange = '30d', limit = 50 } = req.query;
    const leaderboard = await userService.getUserLeaderboard(timeRange, parseInt(limit));

    res.json({
      success: true,
      data: leaderboard
    });

  } catch (error) {
    logger.error('Failed to get leaderboard:', error);
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
});

// Get user by ID (public profile)
router.get('/:userId', async (req, res) => {
  try {
    const { userId } = req.params;
    const user = await userService.getUserById(userId);

    res.json({
      success: true,
      data: user.getPublicProfile()
    });

  } catch (error) {
    logger.error('Failed to get user:', error);
    res.status(404).json({
      success: false,
      message: error.message
    });
  }
});

// Delete user account
router.delete('/account', async (req, res) => {
  try {
    const userId = req.user.userId;
    const { password } = req.body;

    if (!password) {
      return res.status(400).json({
        success: false,
        message: 'Password is required to delete account'
      });
    }

    // Verify password
    const user = await userService.getUserById(userId);
    const isPasswordValid = await user.comparePassword(password);
    
    if (!isPasswordValid) {
      return res.status(401).json({
        success: false,
        message: 'Invalid password'
      });
    }

    await userService.deleteUser(userId);

    res.json({
      success: true,
      message: 'Account deleted successfully'
    });

  } catch (error) {
    logger.error('Failed to delete user account:', error);
    res.status(400).json({
      success: false,
      message: error.message
    });
  }
});

module.exports = router;
