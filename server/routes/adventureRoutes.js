const express = require('express');
const Joi = require('joi');
const adventureService = require('../services/adventureService');
const logger = require('../utils/logger');

const router = express.Router();

// Validation schemas
const generateAdventureSchema = Joi.object({
  location: Joi.object({
    lat: Joi.number().min(-90).max(90).required(),
    lng: Joi.number().min(-180).max(180).required(),
    address: Joi.string().optional()
  }).required(),
  interests: Joi.array().items(Joi.string()).min(1).required(),
  duration: Joi.number().min(30).max(90).optional(),
  maxDistance: Joi.number().min(1000).max(20000).optional(),
  startTime: Joi.date().optional(),
  endTime: Joi.date().optional(),
  socialMode: Joi.string().valid('solo', 'friends', 'meet_new_people').optional(),
  friends: Joi.array().items(Joi.string()).optional()
});

const updateStepSchema = Joi.object({
  stepIndex: Joi.number().min(0).required(),
  action: Joi.string().valid('complete', 'skip', 'start').required()
});

const feedbackSchema = Joi.object({
  rating: Joi.number().min(1).max(5).required(),
  comments: Joi.string().max(500).optional(),
  completedSteps: Joi.array().items(Joi.string()).optional(),
  skippedSteps: Joi.array().items(Joi.string()).optional(),
  suggestions: Joi.string().max(500).optional()
});

const shareAdventureSchema = Joi.object({
  friends: Joi.array().items(Joi.string()).optional(),
  isPublic: Joi.boolean().optional()
});

// Generate new adventure
router.post('/generate', async (req, res) => {
  try {
    const userId = req.user.userId;
    
    // Validate request data
    const { error, value } = generateAdventureSchema.validate(req.body);
    if (error) {
      return res.status(400).json({
        success: false,
        message: 'Validation error',
        errors: error.details.map(detail => detail.message)
      });
    }

    // Generate adventure
    const adventure = await adventureService.generateAdventure(userId, value);

    // Send real-time notification
    if (req.io) {
      req.io.to(`user_${userId}`).emit('adventure_generated', {
        adventureId: adventure._id,
        name: adventure.name,
        message: 'Your adventure is ready!'
      });
    }

    res.status(201).json({
      success: true,
      message: 'Adventure generated successfully',
      data: adventure
    });

  } catch (error) {
    logger.error('Adventure generation failed:', error);
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
});

// Get specific adventure
router.get('/:adventureId', async (req, res) => {
  try {
    const userId = req.user.userId;
    const { adventureId } = req.params;

    const adventure = await adventureService.getAdventure(adventureId, userId);

    res.json({
      success: true,
      data: adventure
    });

  } catch (error) {
    logger.error('Failed to get adventure:', error);
    res.status(404).json({
      success: false,
      message: error.message
    });
  }
});

// Get user's adventures
router.get('/', async (req, res) => {
  try {
    const userId = req.user.userId;
    const { status, limit = 20, offset = 0 } = req.query;

    const adventures = await adventureService.getUserAdventures(
      userId, 
      status, 
      parseInt(limit), 
      parseInt(offset)
    );

    res.json({
      success: true,
      data: adventures
    });

  } catch (error) {
    logger.error('Failed to get user adventures:', error);
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
});

// Start adventure
router.post('/:adventureId/start', async (req, res) => {
  try {
    const userId = req.user.userId;
    const { adventureId } = req.params;

    const adventure = await adventureService.startAdventure(adventureId, userId);

    res.json({
      success: true,
      message: 'Adventure started successfully',
      data: adventure
    });

  } catch (error) {
    logger.error('Failed to start adventure:', error);
    res.status(400).json({
      success: false,
      message: error.message
    });
  }
});

// Complete adventure
router.post('/:adventureId/complete', async (req, res) => {
  try {
    const userId = req.user.userId;
    const { adventureId } = req.params;

    // Validate feedback if provided
    let feedback = null;
    if (req.body.feedback) {
      const { error, value } = feedbackSchema.validate(req.body.feedback);
      if (error) {
        return res.status(400).json({
          success: false,
          message: 'Validation error',
          errors: error.details.map(detail => detail.message)
        });
      }
      feedback = value;
    }

    const adventure = await adventureService.completeAdventure(adventureId, userId, feedback);

    res.json({
      success: true,
      message: 'Adventure completed successfully',
      data: adventure
    });

  } catch (error) {
    logger.error('Failed to complete adventure:', error);
    res.status(400).json({
      success: false,
      message: error.message
    });
  }
});

// Update adventure step
router.put('/:adventureId/steps', async (req, res) => {
  try {
    const userId = req.user.userId;
    const { adventureId } = req.params;

    // Validate request data
    const { error, value } = updateStepSchema.validate(req.body);
    if (error) {
      return res.status(400).json({
        success: false,
        message: 'Validation error',
        errors: error.details.map(detail => detail.message)
      });
    }

    const adventure = await adventureService.updateAdventureStep(
      adventureId, 
      value.stepIndex, 
      userId, 
      value.action
    );

    res.json({
      success: true,
      message: 'Step updated successfully',
      data: adventure
    });

  } catch (error) {
    logger.error('Failed to update adventure step:', error);
    res.status(400).json({
      success: false,
      message: error.message
    });
  }
});

// Get adventure recommendations
router.get('/recommendations', async (req, res) => {
  try {
    const userId = req.user.userId;
    const { lat, lng, limit = 5 } = req.query;

    if (!lat || !lng) {
      return res.status(400).json({
        success: false,
        message: 'Location (lat, lng) is required'
      });
    }

    const location = { lat: parseFloat(lat), lng: parseFloat(lng) };
    const recommendations = await adventureService.getAdventureRecommendations(
      userId, 
      location, 
      parseInt(limit)
    );

    res.json({
      success: true,
      data: recommendations
    });

  } catch (error) {
    logger.error('Failed to get adventure recommendations:', error);
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
});

// Share adventure
router.post('/:adventureId/share', async (req, res) => {
  try {
    const userId = req.user.userId;
    const { adventureId } = req.params;

    // Validate request data
    const { error, value } = shareAdventureSchema.validate(req.body);
    if (error) {
      return res.status(400).json({
        success: false,
        message: 'Validation error',
        errors: error.details.map(detail => detail.message)
      });
    }

    const adventure = await adventureService.shareAdventure(adventureId, userId, value);

    res.json({
      success: true,
      message: 'Adventure shared successfully',
      data: adventure
    });

  } catch (error) {
    logger.error('Failed to share adventure:', error);
    res.status(400).json({
      success: false,
      message: error.message
    });
  }
});

// Get adventure analytics
router.get('/analytics', async (req, res) => {
  try {
    const userId = req.user.userId;
    const { timeRange = '30d' } = req.query;

    const analytics = await adventureService.getAdventureAnalytics(userId, timeRange);

    res.json({
      success: true,
      data: analytics
    });

  } catch (error) {
    logger.error('Failed to get adventure analytics:', error);
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
});

// Cancel adventure
router.post('/:adventureId/cancel', async (req, res) => {
  try {
    const userId = req.user.userId;
    const { adventureId } = req.params;

    const adventure = await adventureService.cancelAdventure(adventureId, userId);

    res.json({
      success: true,
      message: 'Adventure cancelled successfully',
      data: adventure
    });

  } catch (error) {
    logger.error('Failed to cancel adventure:', error);
    res.status(400).json({
      success: false,
      message: error.message
    });
  }
});

// Pause adventure
router.post('/:adventureId/pause', async (req, res) => {
  try {
    const userId = req.user.userId;
    const { adventureId } = req.params;

    const adventure = await adventureService.pauseAdventure(adventureId, userId);

    res.json({
      success: true,
      message: 'Adventure paused successfully',
      data: adventure
    });

  } catch (error) {
    logger.error('Failed to pause adventure:', error);
    res.status(400).json({
      success: false,
      message: error.message
    });
  }
});

// Resume adventure
router.post('/:adventureId/resume', async (req, res) => {
  try {
    const userId = req.user.userId;
    const { adventureId } = req.params;

    const adventure = await adventureService.resumeAdventure(adventureId, userId);

    res.json({
      success: true,
      message: 'Adventure resumed successfully',
      data: adventure
    });

  } catch (error) {
    logger.error('Failed to resume adventure:', error);
    res.status(400).json({
      success: false,
      message: error.message
    });
  }
});

// Get adventure variations
router.get('/:adventureId/variations', async (req, res) => {
  try {
    const userId = req.user.userId;
    const { adventureId } = req.params;

    const variations = await adventureService.getAdventureVariations(adventureId, userId);

    res.json({
      success: true,
      data: variations
    });

  } catch (error) {
    logger.error('Failed to get adventure variations:', error);
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
});

module.exports = router;
