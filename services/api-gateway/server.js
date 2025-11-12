const express = require('express');
const { createProxyMiddleware } = require('http-proxy-middleware');
const cors = require('cors');
const helmet = require('helmet');
const dotenv = require('dotenv');
const winston = require('winston');

dotenv.config();

const app = express();
const PORT = process.env.PORT || 5000;

const logger = winston.createLogger({
  level: 'info',
  format: winston.format.json(),
  transports: [new winston.transports.Console({ format: winston.format.simple() })]
});

app.use(helmet());
app.use(cors());
app.use(express.json());

// Service URLs
const SERVICES = {
  user: process.env.USER_SERVICE_URL || 'http://user-service:5001',
  task: process.env.TASK_SERVICE_URL || 'http://task-service:5002',
  gamification: process.env.GAMIFICATION_SERVICE_URL || 'http://gamification-service:5003',
  analytics: process.env.ANALYTICS_SERVICE_URL || 'http://analytics-service:5004',
  notification: process.env.NOTIFICATION_SERVICE_URL || 'http://notification-service:5005',
  integration: process.env.INTEGRATION_SERVICE_URL || 'http://integration-service:5006',
  challenge: process.env.CHALLENGE_SERVICE_URL || 'http://challenge-service:5007',
  websocket: process.env.WEBSOCKET_SERVICE_URL || 'http://websocket-service:5008',
  pyagent: process.env.PYAGENT_URL || 'http://pyagent:8000'
};

// Health check
app.get('/health', (req, res) => {
  res.json({ 
    status: 'healthy', 
    service: 'api-gateway',
    services: Object.keys(SERVICES),
    timestamp: new Date().toISOString() 
  });
});

// Proxy routes
app.use('/api/users', createProxyMiddleware({
  target: SERVICES.user,
  changeOrigin: true,
  pathRewrite: { '^/api/users': '' },
  onError: (err, req, res) => {
    logger.error('User service proxy error:', err);
    res.status(503).json({ error: 'User service unavailable' });
  }
}));

app.use('/api/tasks', createProxyMiddleware({
  target: SERVICES.task,
  changeOrigin: true,
  pathRewrite: { '^/api/tasks': '/tasks' },
  onError: (err, req, res) => {
    logger.error('Task service proxy error:', err);
    res.status(503).json({ error: 'Task service unavailable' });
  }
}));

app.use('/api/gamification', createProxyMiddleware({
  target: SERVICES.gamification,
  changeOrigin: true,
  pathRewrite: { '^/api/gamification': '' },
  onError: (err, req, res) => {
    logger.error('Gamification service proxy error:', err);
    res.status(503).json({ error: 'Gamification service unavailable' });
  }
}));

app.use('/api/analytics', createProxyMiddleware({
  target: SERVICES.analytics,
  changeOrigin: true,
  pathRewrite: { '^/api/analytics': '' },
  onError: (err, req, res) => {
    logger.error('Analytics service proxy error:', err);
    res.status(503).json({ error: 'Analytics service unavailable' });
  }
}));

app.use('/api/notifications', createProxyMiddleware({
  target: SERVICES.notification,
  changeOrigin: true,
  pathRewrite: { '^/api/notifications': '' },
  onError: (err, req, res) => {
    logger.error('Notification service proxy error:', err);
    res.status(503).json({ error: 'Notification service unavailable' });
  }
}));

app.use('/api/integrations', createProxyMiddleware({
  target: SERVICES.integration,
  changeOrigin: true,
  pathRewrite: { '^/api/integrations': '' },
  onError: (err, req, res) => {
    logger.error('Integration service proxy error:', err);
    res.status(503).json({ error: 'Integration service unavailable' });
  }
}));

app.use('/api/challenges', createProxyMiddleware({
  target: SERVICES.challenge,
  changeOrigin: true,
  pathRewrite: { '^/api/challenges': '' },
  onError: (err, req, res) => {
    logger.error('Challenge service proxy error:', err);
    res.status(503).json({ error: 'Challenge service unavailable' });
  }
}));

app.use('/api/agent', createProxyMiddleware({
  target: SERVICES.pyagent,
  changeOrigin: true,
  pathRewrite: { '^/api/agent': '/agent' },
  onError: (err, req, res) => {
    logger.error('AI service proxy error:', err);
    res.status(503).json({ error: 'AI service unavailable' });
  }
}));

app.listen(PORT, () => {
  logger.info(`API Gateway running on port ${PORT}`);
  logger.info('Proxying to services:', SERVICES);
});

module.exports = app;

