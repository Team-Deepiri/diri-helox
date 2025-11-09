const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const helmet = require('helmet');
const compression = require('compression');
const morgan = require('morgan');
const rateLimit = require('express-rate-limit');
const path = require('path');
const mongoose = require('mongoose');
const { createServer } = require('http');
const { Server } = require('socket.io');
require('dotenv').config();

// Import services
const adventureService = require('./services/adventureService');
const userService = require('./services/userService');
const eventService = require('./services/eventService');
const notificationService = require('./services/notificationService');
const aiOrchestrator = require('./services/aiOrchestrator');
const cacheService = require('./services/cacheService');

// Import routes
const authRoutes = require('./routes/authRoutes');
const userRoutes = require('./routes/userRoutes');
const userItemRoutes = require('./routes/userItemRoutes');
const adventureRoutes = require('./routes/adventureRoutes');
const agentRoutes = require('./routes/agentRoutes');
const eventRoutes = require('./routes/eventRoutes');
const notificationRoutes = require('./routes/notificationRoutes');
const externalRoutes = require('./routes/externalRoutes');
const logsRoutes = require('./routes/logsRoutes');
const taskRoutes = require('./routes/taskRoutes');
const challengeRoutes = require('./routes/challengeRoutes');
const gamificationRoutes = require('./routes/gamificationRoutes');
const analyticsRoutes = require('./routes/analyticsRoutes');
const integrationRoutes = require('./routes/integrationRoutes');

// Import middleware
const authenticateJWT = require('./middleware/authenticateJWT');
const { errorHandler, notFoundHandler, gracefulShutdown } = require('./middleware/errorHandler');
const logger = require('./utils/logger');
const promClient = require('prom-client');
const { v4: uuidv4 } = require('uuid');
const ipFilter = require('./middleware/ipFilter');
const sanitize = require('./middleware/sanitize');
const rateBot = require('./middleware/rateBot');
const auditLogger = require('./middleware/auditLogger');
const swaggerUi = require('swagger-ui-express');
const swaggerJsdoc = require('swagger-jsdoc');

const app = express();
const server = createServer(app);
const io = new Server(server, {
  cors: {
    origin: process.env.CORS_ORIGIN || ["http://localhost:5173", "http://localhost:3000"],
    methods: ["GET", "POST", "PUT", "PATCH", "DELETE"],
    credentials: true
  },
  // Enhanced Socket.IO configuration for real-time updates
  pingTimeout: 60000,
  pingInterval: 25000,
  upgradeTimeout: 30000,
  allowEIO3: true, // Allow Engine.IO v3 clients
  // Enable compression for better performance
  compression: true,
  // Configure transport options
  transports: ['websocket', 'polling'],
  allowUpgrades: true,
  perMessageDeflate: {
    threshold: 1024,
    concurrencyLimit: 10,
    memLevel: 7
  }
});

const PORT = process.env.PORT || 5000;

// CORS (handle before logging/limiting)
const corsAllowedOrigins = [
  'http://localhost:5173',
  'http://localhost:3000',
  process.env.CORS_ORIGIN
].filter(Boolean);
app.use(cors({
  origin: (origin, callback) => {
    if (!origin || corsAllowedOrigins.includes(origin)) return callback(null, true);
    return callback(new Error('Not allowed by CORS'));
  },
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With'],
  exposedHeaders: ['x-request-id']
}));
app.options(/.*/, cors());

// CORS configuration (allow 3000 + 5173 + env)
const allowedOrigins = [
  'http://localhost:5173',
  'http://localhost:3000',
  process.env.CORS_ORIGIN
].filter(Boolean);
app.use(cors({
  origin: (origin, callback) => {
    if (!origin || allowedOrigins.includes(origin)) return callback(null, true);
    return callback(new Error('Not allowed by CORS'));
  },
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With'],
  exposedHeaders: ['x-request-id']
}));
app.options(/.*/, cors());

// Security middleware
app.use(helmet());
app.use(compression());
app.use(ipFilter());
app.use(sanitize());
app.use(rateBot());
app.use(auditLogger());

// Rate limiting (skip preflight)
const limiter = rateLimit({
  windowMs: parseInt(process.env.RATE_LIMIT_WINDOW_MS) || 15 * 60 * 1000, // 15 minutes
  max: parseInt(process.env.RATE_LIMIT_MAX_REQUESTS) || 100, // limit each IP to 100 requests per windowMs
  message: 'Too many requests from this IP, please try again later.',
  standardHeaders: true,
  legacyHeaders: false,
  // Skip preflight, auth routes, and all requests in non-production envs
  skip: (req) => {
    if (req.method === 'OPTIONS') return true;
    if (process.env.NODE_ENV !== 'production') return true;
    // when mounted at '/api/', req.path begins with '/auth' for auth endpoints
    return req.path && req.path.startsWith('/auth');
  }
});
app.use('/api/', limiter);

// Logging
app.use((req, res, next) => {
  req.requestId = uuidv4();
  res.setHeader('x-request-id', req.requestId);
  next();
});
app.use(morgan(':method :url :status :res[content-length] - :response-time ms :req[x-request-id]', {
  stream: { write: message => logger.info(message.trim()) }
}));

// Metrics
const collectDefaultMetrics = promClient.collectDefaultMetrics;
collectDefaultMetrics();
const httpRequestDuration = new promClient.Histogram({
  name: 'http_request_duration_seconds',
  help: 'Duration of HTTP requests in seconds',
  labelNames: ['method', 'route', 'code']
});
app.use((req, res, next) => {
  const end = httpRequestDuration.startTimer();
  res.on('finish', () => {
    end({ method: req.method, route: req.path, code: res.statusCode });
  });
  next();
});

// (CORS already configured above)

// Body parsing middleware
app.use(bodyParser.json({ limit: '10mb' }));
app.use(bodyParser.urlencoded({ extended: true, limit: '10mb' }));

// OpenAPI docs (disabled by default in production)
const swaggerEnabled = process.env.SWAGGER_ENABLED === 'true' || process.env.NODE_ENV !== 'production';
if (swaggerEnabled) {
  const swaggerSpec = swaggerJsdoc({
    definition: {
      openapi: '3.0.0',
      info: { title: 'Deepiri API', version: '3.0.0' }
    },
    apis: []
  });
  app.use('/api-docs', (req, res, next) => {
    res.setHeader('X-Content-Type-Options', 'nosniff');
    res.setHeader('X-Frame-Options', 'DENY');
    res.setHeader('Referrer-Policy', 'no-referrer');
    res.setHeader('Cache-Control', 'no-store');
    next();
  }, swaggerUi.serve, swaggerUi.setup(swaggerSpec, { explorer: false }));
}

// Database connection
const mongoUri = process.env.MONGODB_URI || process.env.MONGO_URI || 'mongodb://localhost:27017/deepiri';
mongoose.connect(mongoUri, {
  // Removed deprecated options: useNewUrlParser and useUnifiedTopology
})
.then(() => {
  logger.info('Connected to MongoDB');
})
.catch((error) => {
  logger.error('MongoDB connection error:', error);
  process.exit(1);
});

// Initialize services
cacheService.initialize();
aiOrchestrator.initialize();

// Socket.IO connection handling
io.on('connection', (socket) => {
  logger.info(`User connected: ${socket.id}`);
  
  // Send immediate connection confirmation
  socket.emit('connection_confirmed', {
    socketId: socket.id,
    timestamp: new Date().toISOString(),
    serverStatus: 'connected'
  });
  
  socket.on('join_user_room', (userId) => {
    socket.join(`user_${userId}`);
    logger.info(`User ${userId} joined their room`);
    socket.emit('room_joined', { room: `user_${userId}`, type: 'user' });
  });
  
  socket.on('join_adventure_room', (adventureId) => {
    socket.join(`adventure_${adventureId}`);
    logger.info(`User joined adventure room: ${adventureId}`);
    socket.emit('room_joined', { room: `adventure_${adventureId}`, type: 'adventure' });
  });

  // Multiplayer Collaboration Handlers
  socket.on('collaboration:join', (data) => {
    const { roomId, userId, userInfo } = data;
    socket.join(`collaboration_${roomId}`);
    socket.to(`collaboration_${roomId}`).emit('collaborator:joined', {
      userId,
      ...userInfo,
      socketId: socket.id,
      joinedAt: new Date().toISOString()
    });
    logger.info(`User ${userId} joined collaboration room ${roomId}`);
  });

  socket.on('collaboration:leave', (data) => {
    const { roomId } = data;
    socket.to(`collaboration_${roomId}`).emit('collaborator:left', {
      socketId: socket.id,
      leftAt: new Date().toISOString()
    });
    socket.leave(`collaboration_${roomId}`);
  });

  socket.on('collaboration:update', (data) => {
    const { roomId, update } = data;
    socket.to(`collaboration_${roomId}`).emit('collaboration:state', update);
  });

  // Duel Handlers
  socket.on('duel:challenge', (data) => {
    const { targetUserId, challengeConfig } = data;
    socket.to(`user_${targetUserId}`).emit('duel:invite', {
      id: `duel_${Date.now()}_${socket.id}`,
      fromUserId: socket.userId || socket.id,
      fromUserName: data.fromUserName || 'Unknown',
      challengeConfig,
      timestamp: Date.now()
    });
    logger.info(`Duel challenge sent from ${socket.id} to ${targetUserId}`);
  });

  socket.on('duel:accept', (data) => {
    const { duelId } = data;
    // Create duel session
    const duelState = {
      id: duelId,
      participants: [
        { userId: socket.userId || socket.id, progress: 0 },
        { userId: data.opponentUserId, progress: 0 }
      ],
      challengeName: data.challengeConfig?.challengeType || 'Duel',
      startTime: Date.now(),
      endTime: Date.now() + (data.challengeConfig?.duration || 15 * 60 * 1000)
    };
    
    socket.emit('duel:start', duelState);
    socket.to(`user_${data.opponentUserId}`).emit('duel:start', duelState);
    logger.info(`Duel ${duelId} started`);
  });

  socket.on('duel:reject', (data) => {
    const { duelId } = data;
    socket.to(`user_${data.fromUserId}`).emit('duel:rejected', { duelId });
  });

  socket.on('duel:progress', (data) => {
    const { duelId, progress } = data;
    socket.to(`duel_${duelId}`).emit('duel:update', {
      duelId,
      userId: socket.userId || socket.id,
      progress
    });
  });

  // Team/Guild Handlers
  socket.on('team:join', (data) => {
    const { teamId, userId } = data;
    socket.join(`team_${teamId}`);
    socket.to(`team_${teamId}`).emit('team:member:joined', {
      userId,
      teamId,
      joinedAt: new Date().toISOString()
    });
    logger.info(`User ${userId} joined team ${teamId}`);
  });

  socket.on('team:leave', (data) => {
    const { teamId } = data;
    socket.to(`team_${teamId}`).emit('team:member:left', {
      userId: socket.userId || socket.id,
      teamId
    });
    socket.leave(`team_${teamId}`);
  });

  socket.on('team:mission:start', (data) => {
    const { teamId, missionConfig } = data;
    const mission = {
      id: `mission_${Date.now()}`,
      name: missionConfig.name || 'Team Mission',
      teamId,
      startTime: Date.now(),
      endTime: Date.now() + (missionConfig.duration || 60 * 60 * 1000),
      overallProgress: 0,
      contributions: []
    };
    io.to(`team_${teamId}`).emit('team:mission:started', mission);
  });

  socket.on('team:mission:progress', (data) => {
    const { teamId, missionId, progress } = data;
    io.to(`team_${teamId}`).emit('team:mission:update', {
      missionId,
      userId: socket.userId || socket.id,
      progress,
      timestamp: Date.now()
    });
  });
  
  // Development mode: Enable file change notifications
  if (process.env.NODE_ENV !== 'production') {
    socket.on('file_changed', (data) => {
      socket.broadcast.emit('file_updated', data);
    });
    
    socket.on('ping', () => {
      socket.emit('pong', { timestamp: new Date().toISOString() });
    });
  }
  
  socket.on('disconnect', (reason) => {
    logger.info(`User disconnected: ${socket.id}, reason: ${reason}`);
  });
  
  socket.on('error', (error) => {
    logger.error(`Socket.IO error for ${socket.id}:`, error);
  });
});

// Make io available to routes
app.use((req, res, next) => {
  req.io = io;
  next();
});

// API Routes
app.use('/api/auth', authRoutes);
app.use('/api/users', authenticateJWT, userRoutes);
app.use('/api/user-items', authenticateJWT, userItemRoutes);
app.use('/api/adventures', authenticateJWT, adventureRoutes);
app.use('/api/events', authenticateJWT, eventRoutes);
app.use('/api/notifications', authenticateJWT, notificationRoutes);
app.use('/api/external', externalRoutes);
app.use('/api/agent', authenticateJWT, agentRoutes);
app.use('/api/logs', logsRoutes);
app.use('/api/tasks', authenticateJWT, taskRoutes);
app.use('/api/challenges', authenticateJWT, challengeRoutes);
app.use('/api/gamification', authenticateJWT, gamificationRoutes);
app.use('/api/analytics', authenticateJWT, analyticsRoutes);
app.use('/api/integrations', authenticateJWT, integrationRoutes);

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    version: '3.0.0',
    services: {
      database: mongoose.connection.readyState === 1 ? 'connected' : 'disconnected',
      cache: cacheService.getConnectionStatus() ? 'connected' : 'disconnected',
      ai: aiOrchestrator.isReady() ? 'ready' : 'initializing'
    }
  });
});

app.get('/metrics', async (req, res) => {
  try {
    res.set('Content-Type', promClient.register.contentType);
    res.end(await promClient.register.metrics());
  } catch (err) {
    res.status(500).end(err.message);
  }
});

// Serve static files from the React app in production
if (process.env.NODE_ENV === 'production') {
  app.use(express.static(path.join(__dirname, '../frontend/dist')));
  
  // Catch all handler: send back React's index.html file
  app.get(/.*/, (req, res) => {
    res.sendFile(path.join(__dirname, '../frontend/dist/index.html'));
  });
}

// 404 handler for unmatched routes
app.use(notFoundHandler);

// Error handling middleware
app.use(errorHandler);

// Graceful shutdown handlers
process.on('SIGTERM', gracefulShutdown(server));
process.on('SIGINT', gracefulShutdown(server));

server.listen(PORT, () => {
  logger.info(`Deepiri Server is running on port ${PORT}`);
  logger.info(`Environment: ${process.env.NODE_ENV || 'development'}`);
});

module.exports = { app, server, io };