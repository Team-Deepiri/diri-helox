/**
 * WebSocket Service
 * Real-time notifications via WebSocket
 */
const { Server } = require('socket.io');
const logger = require('../../utils/logger');

class WebSocketService {
  constructor() {
    this.io = null;
    this.connectedUsers = new Map(); // userId -> socketId[]
    this.userRooms = new Map(); // userId -> Set of room names
  }

  initialize(server) {
    this.io = new Server(server, {
      cors: {
        origin: process.env.FRONTEND_URL || '*',
        methods: ['GET', 'POST']
      },
      transports: ['websocket', 'polling']
    });

    this.io.on('connection', (socket) => {
      this._handleConnection(socket);
    });

    logger.info('WebSocket server initialized');
  }

  _handleConnection(socket) {
    socket.on('authenticate', (data) => {
      const { userId } = data;
      if (userId) {
        this._addUserConnection(userId, socket.id);
        socket.join(`user:${userId}`);
        socket.emit('authenticated', { success: true });
        logger.info('User authenticated via WebSocket', { userId, socketId: socket.id });
      }
    });

    socket.on('disconnect', () => {
      this._removeUserConnection(socket.id);
      logger.info('User disconnected from WebSocket', { socketId: socket.id });
    });

    socket.on('subscribe', (room) => {
      socket.join(room);
      logger.debug('Socket subscribed to room', { socketId: socket.id, room });
    });

    socket.on('unsubscribe', (room) => {
      socket.leave(room);
      logger.debug('Socket unsubscribed from room', { socketId: socket.id, room });
    });
  }

  /**
   * Send notification to user
   */
  sendToUser(userId, notification) {
    if (!this.io) {
      logger.warn('WebSocket server not initialized');
      return;
    }

    this.io.to(`user:${userId}`).emit('notification', notification);
    logger.info('Notification sent via WebSocket', { userId, type: notification.type });
  }

  /**
   * Send to multiple users
   */
  sendToUsers(userIds, notification) {
    if (!this.io) {
      logger.warn('WebSocket server not initialized');
      return;
    }

    userIds.forEach(userId => {
      this.io.to(`user:${userId}`).emit('notification', notification);
    });

    logger.info('Notification sent to multiple users', { count: userIds.length });
  }

  /**
   * Broadcast to room
   */
  broadcastToRoom(room, notification) {
    if (!this.io) {
      logger.warn('WebSocket server not initialized');
      return;
    }

    this.io.to(room).emit('notification', notification);
    logger.info('Notification broadcast to room', { room });
  }

  /**
   * Send challenge update
   */
  sendChallengeUpdate(userId, challengeId, update) {
    this.sendToUser(userId, {
      type: 'challenge_update',
      challengeId,
      ...update
    });
  }

  /**
   * Send real-time progress
   */
  sendProgressUpdate(userId, progress) {
    this.sendToUser(userId, {
      type: 'progress_update',
      ...progress
    });
  }

  _addUserConnection(userId, socketId) {
    if (!this.connectedUsers.has(userId)) {
      this.connectedUsers.set(userId, []);
    }
    this.connectedUsers.get(userId).push(socketId);
  }

  _removeUserConnection(socketId) {
    for (const [userId, socketIds] of this.connectedUsers.entries()) {
      const index = socketIds.indexOf(socketId);
      if (index > -1) {
        socketIds.splice(index, 1);
        if (socketIds.length === 0) {
          this.connectedUsers.delete(userId);
        }
        break;
      }
    }
  }

  getConnectedUsers() {
    return Array.from(this.connectedUsers.keys());
  }

  isUserConnected(userId) {
    return this.connectedUsers.has(userId) && this.connectedUsers.get(userId).length > 0;
  }
}

module.exports = new WebSocketService();

