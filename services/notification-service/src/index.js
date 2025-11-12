const express = require('express');
const router = express.Router();

const websocketService = require('./websocketService');
const pushNotificationService = require('./pushNotificationService');

// Services export instances, not classes
const websocket = websocketService;
const push = pushNotificationService;

// WebSocket routes (for HTTP endpoints)
router.get('/websocket/status', (req, res) => {
  res.json({
    status: 'ok',
    connections: websocket.connectedUsers.size,
    timestamp: new Date().toISOString()
  });
});

// Push notification routes
router.post('/push/send', async (req, res) => {
  try {
    const { userId, deviceToken, notification } = req.body;
    const result = await push.sendPushNotification(userId, deviceToken, notification);
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

router.get('/push/:userId', async (req, res) => {
  try {
    const { userId } = req.params;
    const notifications = await push.getUserNotifications(userId);
    res.json(notifications);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

module.exports = { router, websocket, push };

