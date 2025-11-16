"""
PPO (Proximal Policy Optimization) Agent
For challenge difficulty optimization
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple
from collections import deque
from ..logging_config import get_logger
from .rl_environment import get_rl_environment

logger = get_logger("cyrex.ppo_agent")


class PolicyNetwork(nn.Module):
    """Policy network for PPO."""
    
    def __init__(self, state_dim: int = 5, action_dim: int = 3, hidden_dim: int = 64):
        super().__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_std = nn.Linear(hidden_dim, action_dim)
        
        self.activation = nn.ReLU()
        self.tanh = nn.Tanh()
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning mean and std of action distribution."""
        x = self.activation(self.fc1(state))
        x = self.activation(self.fc2(x))
        
        mean = self.tanh(self.fc_mean(x))
        std = torch.clamp(torch.exp(self.fc_std(x)), 0.01, 1.0)
        
        return mean, std


class ValueNetwork(nn.Module):
    """Value network for PPO."""
    
    def __init__(self, state_dim: int = 5, hidden_dim: int = 64):
        super().__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        self.activation = nn.ReLU()
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass returning state value."""
        x = self.activation(self.fc1(state))
        x = self.activation(self.fc2(x))
        value = self.fc3(x)
        return value


class PPOAgent:
    """
    Proximal Policy Optimization agent for challenge difficulty optimization.
    """
    
    def __init__(
        self,
        state_dim: int = 5,
        action_dim: int = 3,
        lr: float = 3e-4,
        gamma: float = 0.99,
        eps_clip: float = 0.2,
        k_epochs: int = 10,
        device: str = 'cpu'
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.device = device
        
        # Networks
        self.policy_net = PolicyNetwork(state_dim, action_dim).to(device)
        self.value_net = ValueNetwork(state_dim).to(device)
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        
        # Experience buffer
        self.buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'log_probs': [],
            'dones': []
        }
        
        self.env = get_rl_environment()
        
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float]:
        """Select action using policy network."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            mean, std = self.policy_net(state_tensor)
            
            if deterministic:
                action = mean
            else:
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
                action = torch.clamp(action, -1.0, 1.0)
            
            # Convert to numpy
            action_np = action.cpu().numpy().flatten()
            
            # Calculate log probability
            dist = torch.distributions.Normal(mean, std)
            log_prob = dist.log_prob(action).sum(dim=1).item()
            
            return action_np, log_prob
    
    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        log_prob: float,
        done: bool
    ):
        """Store transition in buffer."""
        self.buffer['states'].append(state)
        self.buffer['actions'].append(action)
        self.buffer['rewards'].append(reward)
        self.buffer['log_probs'].append(log_prob)
        self.buffer['dones'].append(done)
    
    def update(self):
        """Update policy using PPO algorithm."""
        if len(self.buffer['states']) < 10:
            return  # Need minimum samples
        
        # Convert to tensors
        states = torch.FloatTensor(self.buffer['states']).to(self.device)
        actions = torch.FloatTensor(self.buffer['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(self.buffer['log_probs']).to(self.device)
        
        # Calculate returns
        rewards = self.buffer['rewards']
        dones = self.buffer['dones']
        returns = self._calculate_returns(rewards, dones)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        # Normalize returns
        returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)
        
        # Calculate advantages
        with torch.no_grad():
            values = self.value_net(states).squeeze()
        advantages = returns_tensor - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(self.k_epochs):
            # Get current policy
            mean, std = self.policy_net(states)
            dist = torch.distributions.Normal(mean, std)
            new_log_probs = dist.log_prob(actions).sum(dim=1)
            entropy = dist.entropy().sum(dim=1).mean()
            
            # Calculate ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # PPO clipped objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy
            
            # Value loss
            values = self.value_net(states).squeeze()
            value_loss = nn.MSELoss()(values, returns_tensor)
            
            # Update networks
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
        
        # Clear buffer
        self._clear_buffer()
        
        logger.info('PPO agent updated')
    
    def _calculate_returns(self, rewards: List[float], dones: List[bool]) -> List[float]:
        """Calculate discounted returns."""
        returns = []
        G = 0
        
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                G = 0
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        return returns
    
    def _clear_buffer(self):
        """Clear experience buffer."""
        self.buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'log_probs': [],
            'dones': []
        }
    
    def save_model(self, path: str):
        """Save model to file."""
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'value_state_dict': self.value_net.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict()
        }, path)
        logger.info('PPO model saved', path=path)
    
    def load_model(self, path: str):
        """Load model from file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        logger.info('PPO model loaded', path=path)


# Singleton instance
_ppo_agent = None

def get_ppo_agent() -> PPOAgent:
    """Get singleton PPO agent instance."""
    global _ppo_agent
    if _ppo_agent is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _ppo_agent = PPOAgent(device=device)
    return _ppo_agent


