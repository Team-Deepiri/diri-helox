"""
Reinforcement Learning Environment
OpenAI Gym compatible environment for challenge optimization
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple, Optional
from ..logging_config import get_logger

logger = get_logger("cyrex.rl_environment")


class ChallengeOptimizationEnv(gym.Env):
    """
    OpenAI Gym compatible environment for challenge difficulty optimization.
    
    State: User cognitive state, task complexity, historical performance
    Action: Challenge parameters (difficulty, time_limit, reward_multiplier)
    Reward: User engagement and completion rate
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self):
        super().__init__()
        
        # Action space: [difficulty, time_limit, reward_multiplier]
        # difficulty: 0.0-1.0 (normalized)
        # time_limit: 5-120 minutes (normalized to 0.0-1.0)
        # reward_multiplier: 0.5-2.0 (normalized to 0.0-1.0)
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # State space: [user_skill, task_complexity, recent_performance, time_of_day, energy_level]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        self.state = None
        self.user_state = {}
        self.episode_history = []
        
    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # Initialize random state
        self.state = self.observation_space.sample()
        self.episode_history = []
        
        return self.state, {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: [difficulty, time_limit, reward_multiplier]
            
        Returns:
            observation: New state
            reward: Reward for this step
            terminated: Whether episode is done
            truncated: Whether episode was truncated
            info: Additional information
        """
        # Denormalize action
        difficulty = action[0]
        time_limit = 5 + (action[1] * 115)  # 5-120 minutes
        reward_multiplier = 0.5 + (action[2] * 1.5)  # 0.5-2.0
        
        # Simulate challenge outcome
        completion_rate = self._simulate_completion(difficulty, time_limit)
        engagement_score = self._calculate_engagement(difficulty, time_limit, reward_multiplier)
        
        # Calculate reward
        reward = self._calculate_reward(completion_rate, engagement_score, action)
        
        # Update state
        self.state = self._update_state(completion_rate, engagement_score)
        
        # Check if episode is done
        terminated = len(self.episode_history) >= 10  # 10 challenges per episode
        truncated = False
        
        # Store history
        self.episode_history.append({
            'action': action,
            'completion_rate': completion_rate,
            'engagement': engagement_score,
            'reward': reward
        })
        
        info = {
            'completion_rate': completion_rate,
            'engagement': engagement_score,
            'difficulty': difficulty,
            'time_limit': time_limit
        }
        
        return self.state, reward, terminated, truncated, info
    
    def _simulate_completion(self, difficulty: float, time_limit: float) -> float:
        """Simulate challenge completion rate based on difficulty and time."""
        user_skill = self.state[0]
        task_complexity = self.state[1]
        
        # Optimal difficulty matches user skill
        difficulty_match = 1.0 - abs(difficulty - user_skill)
        
        # Time adequacy
        required_time = task_complexity * 60  # minutes
        time_adequacy = min(1.0, time_limit / max(required_time, 1))
        
        # Base completion rate
        base_rate = 0.5 + (difficulty_match * 0.3) + (time_adequacy * 0.2)
        
        # Add some noise
        noise = np.random.normal(0, 0.1)
        completion_rate = np.clip(base_rate + noise, 0.0, 1.0)
        
        return completion_rate
    
    def _calculate_engagement(self, difficulty: float, time_limit: float, reward_multiplier: float) -> float:
        """Calculate engagement score."""
        user_skill = self.state[0]
        energy_level = self.state[4]
        
        # Challenge should be slightly above user skill for engagement
        optimal_difficulty = user_skill + 0.1
        difficulty_engagement = 1.0 - abs(difficulty - optimal_difficulty)
        
        # Reward multiplier increases engagement
        reward_engagement = reward_multiplier * 0.3
        
        # Energy affects engagement
        energy_factor = energy_level * 0.2
        
        engagement = difficulty_engagement * 0.5 + reward_engagement + energy_factor
        return np.clip(engagement, 0.0, 1.0)
    
    def _calculate_reward(self, completion_rate: float, engagement: float, action: np.ndarray) -> float:
        """Calculate reward signal."""
        # Primary reward: completion rate
        completion_reward = completion_rate * 10
        
        # Secondary reward: engagement
        engagement_reward = engagement * 5
        
        # Penalty for extreme difficulty (too easy or too hard)
        difficulty = action[0]
        difficulty_penalty = -abs(difficulty - 0.5) * 2
        
        # Penalty for very short or very long time limits
        time_limit_norm = action[1]
        time_penalty = -abs(time_limit_norm - 0.5) * 1
        
        total_reward = completion_reward + engagement_reward + difficulty_penalty + time_penalty
        
        return float(total_reward)
    
    def _update_state(self, completion_rate: float, engagement: float) -> np.ndarray:
        """Update state based on outcomes."""
        # Update user skill based on completion
        skill_increase = completion_rate * 0.01
        new_skill = np.clip(self.state[0] + skill_increase, 0.0, 1.0)
        
        # Update recent performance
        new_performance = (self.state[2] * 0.9) + (completion_rate * 0.1)
        
        # Keep other state components (would update based on real data)
        new_state = np.array([
            new_skill,
            self.state[1],  # task_complexity (unchanged)
            new_performance,
            self.state[3],  # time_of_day (unchanged)
            max(0.0, self.state[4] - 0.1)  # energy decreases
        ])
        
        return new_state.astype(np.float32)
    
    def render(self, mode='human'):
        """Render environment state."""
        if mode == 'human':
            print(f"State: {self.state}")
            if self.episode_history:
                last = self.episode_history[-1]
                print(f"Last action: {last['action']}")
                print(f"Completion: {last['completion_rate']:.2f}, Engagement: {last['engagement']:.2f}")
    
    def set_user_state(self, user_state: Dict):
        """Set user state for personalization."""
        self.user_state = user_state
        if 'skill_level' in user_state:
            self.state[0] = user_state['skill_level']
        if 'recent_performance' in user_state:
            self.state[2] = user_state['recent_performance']


# Singleton instance
_env = None

def get_rl_environment() -> ChallengeOptimizationEnv:
    """Get singleton RL environment instance."""
    global _env
    if _env is None:
        _env = ChallengeOptimizationEnv()
    return _env


