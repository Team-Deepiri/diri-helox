const { createServer } = require('http');
const { Server } = require('socket.io');
const express = require('express');
const cors = require('cors');
const dotenv = require('dotenv');
const winston = require('winston');

dotenv.config();

const app = express();
const httpServer = createServer(app);
const io = new Server(httpServer, {
  cors: { origin: '*' }
});

const PORT = process.env.PORT || 5008;

const logger = winston.createLogger({
  level: 'info',
  format: winston.format.json(),
  transports: [new winston.transports.Console({ format: winston.format.simple() })]
});

app.use(cors());
app.use(express.json());

// Socket.IO connection handling
io.on('connection', (socket) => {
  logger.info(`WebSocket client connected: ${socket.id}`);
  
  socket.emit('connection_confirmed', {
    socketId: socket.id,
    timestamp: new Date().toISOString()
  });
  
  socket.on('join_user_room', (userId) => {
    socket.join(`user_${userId}`);
    logger.info(`User ${userId} joined room`);
  });
  
  socket.on('join_adventure_room', (adventureId) => {
    socket.join(`adventure_${adventureId}`);
    logger.info(`User joined adventure room: ${adventureId}`);
  });
  
  socket.on('disconnect', (reason) => {
    logger.info(`WebSocket client disconnected: ${socket.id}, reason: ${reason}`);
  });
});

app.get('/health', (req, res) => {
  res.json({ 
    status: 'healthy', 
    service: 'websocket-service',
    connections: io.sockets.sockets.size,
    timestamp: new Date().toISOString() 
  });
});

httpServer.listen(PORT, () => {
  logger.info(`WebSocket Service running on port ${PORT}`);
});

module.exports = { app, io };

