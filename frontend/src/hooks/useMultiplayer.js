import { useState, useEffect, useCallback } from 'react';
import multiplayerService from '../services/multiplayerService';

/**
 * Hook for managing multiplayer state
 */
export function useMultiplayer() {
  const [connected, setConnected] = useState(false);
  const [collaborators, setCollaborators] = useState([]);
  const [duelState, setDuelState] = useState(null);
  const [teamState, setTeamState] = useState(null);

  useEffect(() => {
    const userId = localStorage.getItem('user_id');
    const token = localStorage.getItem('token');

    if (userId && token) {
      multiplayerService.connect(userId, token);
    }

    // Set up event listeners
    const handleConnected = () => setConnected(true);
    const handleDisconnected = () => {
      setConnected(false);
      setCollaborators([]);
    };
    const handleCollaboratorUpdate = (collabs) => setCollaborators(collabs);
    const handleDuelStarted = (duel) => setDuelState(duel);
    const handleDuelProgress = (duel) => setDuelState(duel);
    const handleDuelFinished = () => setDuelState(null);
    const handleTeamUpdate = (team) => setTeamState(team);

    multiplayerService.on('multiplayer:connected', handleConnected);
    multiplayerService.on('disconnect', handleDisconnected);
    multiplayerService.on('collaborator:update', handleCollaboratorUpdate);
    multiplayerService.on('duel:started', handleDuelStarted);
    multiplayerService.on('duel:progress', handleDuelProgress);
    multiplayerService.on('duel:finished', handleDuelFinished);
    multiplayerService.on('team:update', handleTeamUpdate);

    return () => {
      multiplayerService.off('multiplayer:connected', handleConnected);
      multiplayerService.off('disconnect', handleDisconnected);
      multiplayerService.off('collaborator:update', handleCollaboratorUpdate);
      multiplayerService.off('duel:started', handleDuelStarted);
      multiplayerService.off('duel:progress', handleDuelProgress);
      multiplayerService.off('duel:finished', handleDuelFinished);
      multiplayerService.off('team:update', handleTeamUpdate);
      multiplayerService.disconnect();
    };
  }, []);

  const joinRoom = useCallback((roomId, userId, userInfo) => {
    multiplayerService.joinCollaborationRoom(roomId, userId, userInfo);
  }, []);

  const leaveRoom = useCallback(() => {
    multiplayerService.leaveCollaborationRoom();
  }, []);

  const sendUpdate = useCallback((update) => {
    multiplayerService.sendCollaborationUpdate(update);
  }, []);

  const challengeToDuel = useCallback((targetUserId, challengeConfig) => {
    multiplayerService.challengeToDuel(targetUserId, challengeConfig);
  }, []);

  const acceptDuel = useCallback((duelId) => {
    multiplayerService.acceptDuel(duelId);
  }, []);

  const rejectDuel = useCallback((duelId) => {
    multiplayerService.rejectDuel(duelId);
  }, []);

  const updateDuelProgress = useCallback((progress) => {
    multiplayerService.updateDuelProgress(progress);
  }, []);

  const joinTeam = useCallback((teamId, userId) => {
    multiplayerService.joinTeam(teamId, userId);
  }, []);

  const leaveTeam = useCallback((teamId) => {
    multiplayerService.leaveTeam(teamId);
  }, []);

  const startTeamMission = useCallback((teamId, missionConfig) => {
    multiplayerService.startTeamMission(teamId, missionConfig);
  }, []);

  const updateTeamMissionProgress = useCallback((teamId, missionId, progress) => {
    multiplayerService.updateTeamMissionProgress(teamId, missionId, progress);
  }, []);

  return {
    connected,
    collaborators,
    duelState,
    teamState,
    joinRoom,
    leaveRoom,
    sendUpdate,
    challengeToDuel,
    acceptDuel,
    rejectDuel,
    updateDuelProgress,
    joinTeam,
    leaveTeam,
    startTeamMission,
    updateTeamMissionProgress
  };
}

