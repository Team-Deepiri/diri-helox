const express = require('express');
const Joi = require('joi');
const agentService = require('../services/agentService');
const logger = require('../utils/logger');

const router = express.Router();

const createSessionSchema = Joi.object({
  title: Joi.string().max(120).optional(),
  settings: Joi.object({
    model: Joi.string().optional(),
    temperature: Joi.number().min(0).max(2).optional(),
    topP: Joi.number().min(0).max(1).optional()
  }).optional()
});

const sendMessageSchema = Joi.object({
  content: Joi.string().min(1).required()
});

router.post('/sessions', async (req, res) => {
  try {
    const userId = req.user.userId;
    const { error, value } = createSessionSchema.validate(req.body || {});
    if (error) {
      return res.status(400).json({ success: false, message: 'Validation error', errors: error.details.map(d => d.message) });
    }
    const session = await agentService.createSession(userId, value.title, value.settings);
    res.status(201).json({ success: true, data: session });
  } catch (err) {
    logger.error('Failed to create agent session:', err);
    res.status(500).json({ success: false, message: err.message });
  }
});

router.get('/sessions', async (req, res) => {
  try {
    const userId = req.user.userId;
    const { limit = 20, offset = 0 } = req.query;
    const sessions = await agentService.listSessions(userId, parseInt(limit), parseInt(offset));
    res.json({ success: true, data: sessions });
  } catch (err) {
    logger.error('Failed to list sessions:', err);
    res.status(500).json({ success: false, message: err.message });
  }
});

router.post('/sessions/:sessionId/messages', async (req, res) => {
  try {
    const userId = req.user.userId;
    const { sessionId } = req.params;
    const { error, value } = sendMessageSchema.validate(req.body);
    if (error) {
      return res.status(400).json({ success: false, message: 'Validation error', errors: error.details.map(d => d.message) });
    }

    const result = await agentService.sendMessage(sessionId, userId, value.content);
    res.json({ success: true, data: result });
  } catch (err) {
    logger.error('Failed to send agent message:', err);
    res.status(400).json({ success: false, message: err.message });
  }
});

router.post('/sessions/:sessionId/archive', async (req, res) => {
  try {
    const userId = req.user.userId;
    const { sessionId } = req.params;
    const session = await agentService.archiveSession(sessionId, userId);
    res.json({ success: true, data: session });
  } catch (err) {
    logger.error('Failed to archive session:', err);
    res.status(400).json({ success: false, message: err.message });
  }
});

module.exports = router;


