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
  auth: process.env.AUTH_SERVICE_URL || 'http://auth-service:5001',
  task: process.env.TASK_ORCHESTRATOR_URL || 'http://task-orchestrator:5002',
  engagement: process.env.ENGAGEMENT_SERVICE_URL || 'http://engagement-service:5003',
  analytics: process.env.PLATFORM_ANALYTICS_SERVICE_URL || 'http://platform-analytics-service:5004',
  notification: process.env.NOTIFICATION_SERVICE_URL || 'http://notification-service:5005',
  integration: process.env.EXTERNAL_BRIDGE_SERVICE_URL || 'http://external-bridge-service:5006',
  challenge: process.env.CHALLENGE_SERVICE_URL || 'http://challenge-service:5007',
  realtime: process.env.REALTIME_GATEWAY_URL || 'http://realtime-gateway:5008',
  cyrex: process.env.CYREX_URL || 'http://cyrex:8000'
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
  target: SERVICES.auth,
  changeOrigin: true,
  pathRewrite: { '^/api/users': '' },
  onError: (err, req, res) => {
    logger.error('Auth service proxy error:', err);
    res.status(503).json({ error: 'Auth service unavailable' });
  }
}));

app.use('/api/tasks', createProxyMiddleware({
  target: SERVICES.task,
  changeOrigin: true,
  pathRewrite: { '^/api/tasks': '/tasks' },
  onError: (err, req, res) => {
    logger.error('Task orchestrator proxy error:', err);
    res.status(503).json({ error: 'Task orchestrator unavailable' });
  }
}));

app.use('/api/gamification', createProxyMiddleware({
  target: SERVICES.engagement,
  changeOrigin: true,
  pathRewrite: { '^/api/gamification': '' },
  onError: (err, req, res) => {
    logger.error('Engagement service proxy error:', err);
    res.status(503).json({ error: 'Engagement service unavailable' });
  }
}));

app.use('/api/analytics', createProxyMiddleware({
  target: SERVICES.analytics,
  changeOrigin: true,
  pathRewrite: { '^/api/analytics': '' },
  onError: (err, req, res) => {
    logger.error('Platform analytics service proxy error:', err);
    res.status(503).json({ error: 'Platform analytics service unavailable' });
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
    logger.error('External bridge service proxy error:', err);
    res.status(503).json({ error: 'External bridge service unavailable' });
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
  target: SERVICES.cyrex,
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

