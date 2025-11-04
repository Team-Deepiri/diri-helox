const express = require('express');
const router = express.Router();
const challengeService = require('../services/challengeService');
const authenticateJWT = require('../middleware/authenticateJWT');
const logger = require('../utils/logger');

// Generate challenge from task
router.post('/generate', authenticateJWT, async (req, res, next) => {
  try {
    const { taskId } = req.body;
    if (!taskId) {
      return res.status(400).json({ success: false, message: 'Task ID is required' });
    }
    const challenge = await challengeService.generateChallenge(req.user.id, taskId);
    res.status(201).json({ success: true, data: challenge });
  } catch (error) {
    logger.error('Error generating challenge:', error);
    next(error);
  }
});

// Get all challenges for user
router.get('/', authenticateJWT, async (req, res, next) => {
  try {
    const challenges = await challengeService.getUserChallenges(req.user.id, req.query);
    res.json({ success: true, data: challenges });
  } catch (error) {
    logger.error('Error fetching challenges:', error);
    next(error);
  }
});

// Get single challenge
router.get('/:id', authenticateJWT, async (req, res, next) => {
  try {
    const challenge = await challengeService.getChallengeById(req.params.id, req.user.id);
    res.json({ success: true, data: challenge });
  } catch (error) {
    logger.error('Error fetching challenge:', error);
    next(error);
  }
});

// Complete challenge
router.post('/:id/complete', authenticateJWT, async (req, res, next) => {
  try {
    const challenge = await challengeService.completeChallenge(
      req.params.id, 
      req.user.id, 
      req.body
    );
    res.json({ success: true, data: challenge });
  } catch (error) {
    logger.error('Error completing challenge:', error);
    next(error);
  }
});

module.exports = router;

