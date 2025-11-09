/**
 * Multiplayer Collaboration Service
 * Handles real-time collaboration, duels, and team features
 */

import io from 'socket.io-client';

class MultiplayerService {
  constructor() {
    this.socket = null;
    this.connected = false;
    this.currentRoom = null;
    this.collaborators = new Map();
    this.duelState = null;
    this.teamState = null;
    this.listeners = new Map();
  }

  /**
   * Initialize connection
   */
  connect(userId, token) {
    if (this.socket && this.connected) {
      return;
    }

    const serverUrl = import.meta.env.VITE_API_URL || 'http://localhost:5000';
    this.socket = io(serverUrl, {
      auth: { token, userId },
      transports: ['websocket', 'polling']
    });

    this.socket.on('connect', () => {
      this.connected = true;
      this.emit('multiplayer:connected', { userId });
    });

    this.socket.on('disconnect', () => {
      this.connected = false;
      this.collaborators.clear();
    });

    this.socket.on('collaborator:joined', (data) => {
      this.collaborators.set(data.userId, data);
      this.emit('collaborator:update', Array.from(this.collaborators.values()));
    });

    this.socket.on('collaborator:left', (data) => {
      this.collaborators.delete(data.userId);
      this.emit('collaborator:update', Array.from(this.collaborators.values()));
    });

    this.socket.on('collaboration:state', (data) => {
      this.emit('state:update', data);
    });

    this.socket.on('duel:invite', (data) => {
      this.emit('duel:invitation', data);
    });

    this.socket.on('duel:start', (data) => {
      this.duelState = data;
      this.emit('duel:started', data);
    });

    this.socket.on('duel:update', (data) => {
      this.duelState = { ...this.duelState, ...data };
      this.emit('duel:progress', this.duelState);
    });

    this.socket.on('duel:end', (data) => {
      this.emit('duel:finished', data);
      this.duelState = null;
    });

    this.socket.on('team:joined', (data) => {
      this.teamState = data;
      this.emit('team:update', data);
    });

    this.socket.on('team:mission:update', (data) => {
      this.emit('team:mission:progress', data);
    });
  }

  /**
   * Disconnect
   */
  disconnect() {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
      this.connected = false;
      this.collaborators.clear();
    }
  }

  /**
   * Join collaboration room
   */
  joinCollaborationRoom(roomId, userId, userInfo) {
    if (!this.connected) {
      console.error('Not connected to server');
      return;
    }

    this.currentRoom = roomId;
    this.socket.emit('collaboration:join', {
      roomId,
      userId,
      userInfo
    });
  }

  /**
   * Leave collaboration room
   */
  leaveCollaborationRoom() {
    if (this.currentRoom && this.socket) {
      this.socket.emit('collaboration:leave', { roomId: this.currentRoom });
      this.currentRoom = null;
    }
  }

  /**
   * Send collaboration update
   */
  sendCollaborationUpdate(update) {
    if (this.currentRoom && this.socket) {
      this.socket.emit('collaboration:update', {
        roomId: this.currentRoom,
        update
      });
    }
  }

  /**
   * Challenge user to duel
   */
  challengeToDuel(targetUserId, challengeConfig) {
    if (!this.connected) return;

    this.socket.emit('duel:challenge', {
      targetUserId,
      challengeConfig,
      timestamp: Date.now()
    });
  }

  /**
   * Accept duel invitation
   */
  acceptDuel(duelId) {
    if (!this.connected) return;

    this.socket.emit('duel:accept', { duelId });
  }

  /**
   * Reject duel invitation
   */
  rejectDuel(duelId) {
    if (!this.connected) return;

    this.socket.emit('duel:reject', { duelId });
  }

  /**
   * Update duel progress
   */
  updateDuelProgress(progress) {
    if (!this.connected || !this.duelState) return;

    this.socket.emit('duel:progress', {
      duelId: this.duelState.id,
      progress
    });
  }

  /**
   * Join team/guild
   */
  joinTeam(teamId, userId) {
    if (!this.connected) return;

    this.socket.emit('team:join', { teamId, userId });
  }

  /**
   * Leave team
   */
  leaveTeam(teamId) {
    if (!this.connected) return;

    this.socket.emit('team:leave', { teamId });
  }

  /**
   * Start team mission
   */
  startTeamMission(teamId, missionConfig) {
    if (!this.connected) return;

    this.socket.emit('team:mission:start', {
      teamId,
      missionConfig
    });
  }

  /**
   * Update team mission progress
   */
  updateTeamMissionProgress(teamId, missionId, progress) {
    if (!this.connected) return;

    this.socket.emit('team:mission:progress', {
      teamId,
      missionId,
      progress
    });
  }

  /**
   * Get current collaborators
   */
  getCollaborators() {
    return Array.from(this.collaborators.values());
  }

  /**
   * Get current duel state
   */
  getDuelState() {
    return this.duelState;
  }

  /**
   * Get current team state
   */
  getTeamState() {
    return this.teamState;
  }

  /**
   * Event emitter/listener pattern
   */
  on(event, callback) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, []);
    }
    this.listeners.get(event).push(callback);
  }

  off(event, callback) {
    if (this.listeners.has(event)) {
      const callbacks = this.listeners.get(event);
      const index = callbacks.indexOf(callback);
      if (index > -1) {
        callbacks.splice(index, 1);
      }
    }
  }

  emit(event, data) {
    if (this.listeners.has(event)) {
      this.listeners.get(event).forEach(callback => {
        try {
          callback(data);
        } catch (error) {
          console.error(`Error in ${event} listener:`, error);
        }
      });
    }
  }
}

// Export singleton instance
export default new MultiplayerService();

