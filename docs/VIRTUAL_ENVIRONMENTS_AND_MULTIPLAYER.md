# Virtual Environments and Multiplayer Collaboration

## Overview

Deepiri now includes immersive 3D virtual environments and comprehensive multiplayer collaboration features. These systems work together to create engaging, social productivity experiences.

## Virtual Environments

### Features

- **10+ Immersive 3D Environments**
  - Futuristic Race Track (cyberpunk theme)
  - Mysterious Library (mystical theme)
  - Artistic Studio (creative theme)
  - Cyber Code Space (tech theme)
  - Serene Garden (nature theme)
  - Space Station (sci-fi theme)
  - Medieval Castle (fantasy theme)
  - Underwater Lab (ocean theme)
  - Neon City (urban theme)
  - Zen Temple (zen theme)

- **Dynamic Weather Systems**
  - Sunny, Cloudy, Rainy, Foggy, Stormy
  - Real-time weather effects
  - Environment-specific weather bonuses

- **Time of Day System**
  - Day, Sunset, Night, Dawn
  - Dynamic lighting based on time
  - Adaptive visual effects

- **Environment Bonuses**
  - Challenge-specific bonuses (up to 30% XP boost)
  - Theme-based productivity multipliers
  - Unlockable environments through progression

### Implementation

**Service**: `frontend/src/services/virtualEnvironmentService.js`
- Manages environment state
- Handles environment switching
- Calculates bonuses
- Manages weather and time of day

**Component**: `frontend/src/components/VirtualEnvironment.jsx`
- React Three Fiber 3D rendering
- Particle systems
- Weather effects
- Interactive controls

**Hook**: `frontend/src/hooks/useVirtualEnvironment.js`
- React hook for environment state
- Easy integration with components

### Usage

```jsx
import VirtualEnvironment from '../components/VirtualEnvironment';
import { useVirtualEnvironment } from '../hooks/useVirtualEnvironment';

function MyComponent() {
  const { setEnvironment, getEnvironmentBonus } = useVirtualEnvironment();
  
  // Set environment
  setEnvironment('cyber_code_space', {
    weather: 'sunny',
    timeOfDay: 'day'
  });
  
  // Get bonus for challenge type
  const bonus = getEnvironmentBonus('coding_kata'); // Returns 1.3 (30% bonus)
  
  return (
    <VirtualEnvironment 
      environmentId="cyber_code_space"
      onEnvironmentChange={(id) => setEnvironment(id)}
    />
  );
}
```

## Multiplayer Collaboration

### Features

#### 1. Real-Time Collaboration
- **Collaboration Rooms**: Join shared workspaces
- **Live Presence**: See who's working with you
- **State Synchronization**: Real-time updates across collaborators
- **Collaborator Avatars**: Visual representation of team members

#### 2. Productivity Duels
- **Challenge System**: Challenge other users to timed competitions
- **Real-Time Progress**: Live progress tracking
- **Leaderboards**: Compare performance in real-time
- **Invitation System**: Accept/reject duel invitations

#### 3. Team/Guild System
- **Team Missions**: Collaborative team challenges
- **Shared Progress**: Track team progress together
- **Contribution Tracking**: Individual contribution points
- **Team Chat**: Communication within teams

### Implementation

**Service**: `frontend/src/services/multiplayerService.js`
- Socket.IO client management
- Event handling
- State management
- Connection lifecycle

**Component**: `frontend/src/components/MultiplayerCollaboration.jsx`
- Collaboration UI
- Duel progress display
- Team mission tracking
- Invitation handling

**Hook**: `frontend/src/hooks/useMultiplayer.js`
- React hook for multiplayer state
- Easy integration

**Backend**: `api-server/server.js`
- Socket.IO event handlers
- Room management
- Duel state management
- Team coordination

### Usage

```jsx
import MultiplayerCollaboration from '../components/MultiplayerCollaboration';
import { useMultiplayer } from '../hooks/useMultiplayer';

function MyComponent() {
  const { 
    connected, 
    collaborators, 
    joinRoom, 
    sendUpdate 
  } = useMultiplayer();
  
  // Join collaboration room
  useEffect(() => {
    joinRoom('room_123', userId, { name: 'John Doe' });
  }, []);
  
  // Send update
  const handleUpdate = () => {
    sendUpdate({
      type: 'task_completed',
      taskId: 'task_123',
      progress: 100
    });
  };
  
  return (
    <MultiplayerCollaboration
      roomId="room_123"
      currentUserId={userId}
      currentUserName="John Doe"
      mode="collaboration" // or 'duel' or 'team'
    />
  );
}
```

## Integration Example

See `frontend/src/pages/ImmersiveWorkspace.jsx` for a complete example combining:
- Virtual environment selection
- Multiplayer collaboration
- Real-time updates
- Environment bonuses

## Socket.IO Events

### Collaboration Events
- `collaboration:join` - Join a collaboration room
- `collaboration:leave` - Leave a room
- `collaboration:update` - Send state update
- `collaborator:joined` - New collaborator joined
- `collaborator:left` - Collaborator left
- `collaboration:state` - State update received

### Duel Events
- `duel:challenge` - Challenge another user
- `duel:accept` - Accept duel invitation
- `duel:reject` - Reject duel invitation
- `duel:start` - Duel started
- `duel:progress` - Update duel progress
- `duel:update` - Progress update received
- `duel:end` - Duel finished

### Team Events
- `team:join` - Join a team
- `team:leave` - Leave a team
- `team:mission:start` - Start team mission
- `team:mission:progress` - Update mission progress
- `team:mission:update` - Mission progress received
- `team:member:joined` - Team member joined
- `team:member:left` - Team member left

## Dependencies

### Frontend
- `@react-three/fiber` - 3D rendering
- `@react-three/drei` - 3D utilities
- `three` - 3D library
- `socket.io-client` - Real-time communication

### Backend
- `socket.io` - WebSocket server (already included)

## Configuration

### Environment Variables
```env
# Frontend
VITE_API_URL=http://localhost:5000

# Backend (already configured)
# Socket.IO is automatically set up in server.js
```

## Future Enhancements

1. **3D Environment Customization**
   - User-created environments
   - Asset marketplace
   - Custom themes

2. **Advanced Multiplayer**
   - Voice chat integration
   - Screen sharing
   - Collaborative editing

3. **Social Features**
   - Friend system
   - Guild management
   - Achievement sharing

4. **Performance Optimization**
   - LOD (Level of Detail) for 3D models
   - WebGL optimization
   - Network optimization for multiplayer

---

**Status**: ✅ Fully Implemented
**Last Updated**: 2024

