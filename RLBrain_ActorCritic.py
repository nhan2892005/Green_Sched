import torch
import torch.nn as nn
import torch.optim as optim

# -------------------------------
# Kiến trúc CNN-based Actor-Critic
# -------------------------------
class CNNActor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(CNNActor, self).__init__()
        # Chuyển đổi vector 1D thành (batch, 1, input_dim)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        # Tính kích thước đầu ra sau conv: giả sử input_dim = 524
        conv_output_size = 64 * 64  # ví dụ: 64 channels, chiều dài 64 (tính theo công thức)
        self.fc = nn.Linear(conv_output_size, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        # x có shape (batch, input_dim)
        x = x.unsqueeze(1)  # shape: (batch, 1, input_dim)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc(x))
        x = self.out(x)
        return self.softmax(x)

class CNNCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super(CNNCritic, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        conv_output_size = 64 * 64  # như trên
        self.fc = nn.Linear(conv_output_size, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, input_dim)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc(x))
        value = self.out(x)
        return value

class ActorCriticAgent:
    def __init__(self, input_dim, output_dim, hidden_dim=256, lr=1e-3):
        self.actor = CNNActor(input_dim, output_dim, hidden_dim)
        self.critic = CNNCritic(input_dim, hidden_dim)
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)
        self.log_probs = []
        self.rewards = []
        self.values = []
    
    def select_action(self, state):
        # state: numpy array shape (input_dim,)
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)  # (1, input_dim)
        probs = self.actor(state_tensor)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        self.log_probs.append(m.log_prob(action))
        value = self.critic(state_tensor)
        self.values.append(value)
        return action.item()
    
    def finish_episode(self, gamma=0.99):
        R = 0
        returns = []
        for r in self.rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        # (Tùy chọn) Chuẩn hóa returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        
        log_probs = torch.stack(self.log_probs)
        values = torch.cat(self.values).squeeze()
        advantages = returns - values
        
        actor_loss = - (log_probs * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()
        loss = actor_loss + critic_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Reset memory
        self.log_probs = []
        self.rewards = []
        self.values = []
