const express = require('express');
const router = express.Router();
const analyticsService = require('../services/analyticsService');
const authenticateJWT = require('../middleware/authenticateJWT');
const logger = require('../utils/logger');

// Get user analytics
router.get('/', authenticateJWT, async (req, res, next) => {
  try {
    const days = parseInt(req.query.days) || 30;
    const analytics = await analyticsService.getUserAnalytics(req.user.id, days);
    res.json({ success: true, data: analytics });
  } catch (error) {
    logger.error('Error fetching analytics:', error);
    next(error);
  }
});

// Get productivity stats
router.get('/stats', authenticateJWT, async (req, res, next) => {
  try {
    const period = req.query.period || 'week';
    const stats = await analyticsService.getProductivityStats(req.user.id, period);
    res.json({ success: true, data: stats });
  } catch (error) {
    logger.error('Error fetching productivity stats:', error);
    next(error);
  }
});

module.exports = router;

