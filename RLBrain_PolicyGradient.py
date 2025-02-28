import torch
import torch.nn as nn
import torch.optim as optim

# --- Policy Network & RL Agent (REINFORCE) ---
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)

class RLPolicyAgent:
    def __init__(self, input_dim, output_dim, hidden_dim=256, lr=1e-3):
        self.policy_net = PolicyNetwork(input_dim, output_dim, hidden_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.log_probs = []
        self.rewards = []
    
    def select_action(self, state):
        state = torch.from_numpy(state).float()
        probs = self.policy_net(state)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        self.log_probs.append(m.log_prob(action))
        return action.item()
    
    def finish_episode(self, gamma=0.99):
        R = 0
        returns = []
        for r in self.rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        loss = 0
        for log_prob, R in zip(self.log_probs, returns):
            loss -= log_prob * R
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.rewards = []
        self.log_probs = []