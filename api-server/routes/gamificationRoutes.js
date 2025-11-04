const express = require('express');
const router = express.Router();
const gamificationService = require('../services/gamificationService');
const authenticateJWT = require('../middleware/authenticateJWT');
const logger = require('../utils/logger');

// Get user gamification profile
router.get('/profile', authenticateJWT, async (req, res, next) => {
  try {
    const profile = await gamificationService.getOrCreateProfile(req.user.id);
    const rank = await gamificationService.getUserRank(req.user.id);
    
    res.json({ 
      success: true, 
      data: {
        ...profile.toObject(),
        rank
      }
    });
  } catch (error) {
    logger.error('Error fetching gamification profile:', error);
    next(error);
  }
});

// Get leaderboard
router.get('/leaderboard', authenticateJWT, async (req, res, next) => {
  try {
    const limit = parseInt(req.query.limit) || 100;
    const period = req.query.period || 'all';
    const leaderboard = await gamificationService.getLeaderboard(limit, period);
    res.json({ success: true, data: leaderboard });
  } catch (error) {
    logger.error('Error fetching leaderboard:', error);
    next(error);
  }
});

// Get user rank
router.get('/rank', authenticateJWT, async (req, res, next) => {
  try {
    const rank = await gamificationService.getUserRank(req.user.id);
    res.json({ success: true, data: { rank } });
  } catch (error) {
    logger.error('Error fetching rank:', error);
    next(error);
  }
});

// Check and award badges
router.post('/badges/check', authenticateJWT, async (req, res, next) => {
  try {
    const awardedBadges = await gamificationService.checkAndAwardBadges(req.user.id);
    res.json({ success: true, data: { awardedBadges } });
  } catch (error) {
    logger.error('Error checking badges:', error);
    next(error);
  }
});

// Update preferences
router.patch('/preferences', authenticateJWT, async (req, res, next) => {
  try {
    const profile = await gamificationService.updatePreferences(req.user.id, req.body);
    res.json({ success: true, data: profile });
  } catch (error) {
    logger.error('Error updating preferences:', error);
    next(error);
  }
});

module.exports = router;

