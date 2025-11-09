import React, { useState } from 'react';
import VirtualEnvironment from '../components/VirtualEnvironment';
import MultiplayerCollaboration from '../components/MultiplayerCollaboration';
import { useVirtualEnvironment } from '../hooks/useVirtualEnvironment';
import './ImmersiveWorkspace.css';

/**
 * Immersive Workspace Page
 * Combines 3D virtual environments with multiplayer collaboration
 */
export default function ImmersiveWorkspace() {
  const [selectedEnvironment, setSelectedEnvironment] = useState('cyber_code_space');
  const [collaborationMode, setCollaborationMode] = useState('collaboration');
  const [roomId, setRoomId] = useState(null);
  const { getAvailableEnvironments, setEnvironment, setWeather, setTimeOfDay } = useVirtualEnvironment();
  
  const userId = localStorage.getItem('user_id') || 'default_user';
  const userName = localStorage.getItem('user_name') || 'User';

  const environments = getAvailableEnvironments();

  const handleEnvironmentChange = (envId) => {
    setSelectedEnvironment(envId);
    setEnvironment(envId);
  };

  const handleCreateRoom = () => {
    const newRoomId = `room_${Date.now()}_${userId}`;
    setRoomId(newRoomId);
  };

  return (
    <div className="immersive-workspace">
      <div className="workspace-header">
        <h1>Immersive Workspace</h1>
        <div className="workspace-controls">
          <select 
            value={selectedEnvironment}
            onChange={(e) => handleEnvironmentChange(e.target.value)}
            className="environment-select"
          >
            {environments.map(env => (
              <option key={env.id} value={env.id}>
                {env.name}
              </option>
            ))}
          </select>
          <select
            value={collaborationMode}
            onChange={(e) => setCollaborationMode(e.target.value)}
            className="mode-select"
          >
            <option value="collaboration">Collaboration</option>
            <option value="duel">Duel</option>
            <option value="team">Team</option>
          </select>
          {!roomId && (
            <button onClick={handleCreateRoom} className="create-room-btn">
              Create Room
            </button>
          )}
        </div>
      </div>

      <div className="workspace-content">
        <div className="environment-section">
          <VirtualEnvironment 
            environmentId={selectedEnvironment}
            onEnvironmentChange={handleEnvironmentChange}
            className="main-environment"
          />
        </div>

        <div className="collaboration-section">
          {roomId ? (
            <MultiplayerCollaboration
              roomId={roomId}
              currentUserId={userId}
              currentUserName={userName}
              mode={collaborationMode}
            />
          ) : (
            <div className="no-room-message">
              <p>Create or join a room to start collaborating</p>
              <button onClick={handleCreateRoom} className="create-room-btn">
                Create Room
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

