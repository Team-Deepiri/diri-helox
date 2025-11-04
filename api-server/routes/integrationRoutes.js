const express = require('express');
const router = express.Router();
const integrationService = require('../services/integrationService');
const authenticateJWT = require('../middleware/authenticateJWT');
const logger = require('../utils/logger');

// Get user integrations
router.get('/', authenticateJWT, async (req, res, next) => {
  try {
    const integrations = await integrationService.getUserIntegrations(req.user.id);
    res.json({ success: true, data: integrations });
  } catch (error) {
    logger.error('Error fetching integrations:', error);
    next(error);
  }
});

// Connect integration
router.post('/connect', authenticateJWT, async (req, res, next) => {
  try {
    const { service, credentials } = req.body;
    if (!service || !credentials) {
      return res.status(400).json({ 
        success: false, 
        message: 'Service and credentials are required' 
      });
    }
    const integration = await integrationService.connectIntegration(
      req.user.id, 
      service, 
      credentials
    );
    res.status(201).json({ success: true, data: integration });
  } catch (error) {
    logger.error('Error connecting integration:', error);
    next(error);
  }
});

// Disconnect integration
router.post('/:service/disconnect', authenticateJWT, async (req, res, next) => {
  try {
    const integration = await integrationService.disconnectIntegration(
      req.user.id, 
      req.params.service
    );
    res.json({ success: true, data: integration });
  } catch (error) {
    logger.error('Error disconnecting integration:', error);
    next(error);
  }
});

// Sync integration
router.post('/:service/sync', authenticateJWT, async (req, res, next) => {
  try {
    const result = await integrationService.syncIntegration(
      req.user.id, 
      req.params.service
    );
    res.json({ success: true, data: result });
  } catch (error) {
    logger.error('Error syncing integration:', error);
    next(error);
  }
});

// Sync all integrations
router.post('/sync/all', authenticateJWT, async (req, res, next) => {
  try {
    const results = await integrationService.syncAllIntegrations(req.user.id);
    res.json({ success: true, data: results });
  } catch (error) {
    logger.error('Error syncing all integrations:', error);
    next(error);
  }
});

module.exports = router;

