import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

from Env import HPCEnv
from RLBrain import RLPolicyAgent

# -------------------------------
# Cấu hình huấn luyện
# -------------------------------
num_training_episodes = 1024
save_interval = 32

# Tạo thư mục lưu weights và dữ liệu
os.makedirs("weights", exist_ok=True)
os.makedirs("data", exist_ok=True)

env = HPCEnv()
input_dim = env.observation_space.shape[0]  # 524 features
output_dim = env.action_space.n             # JOB_QUEUE_SIZE (64)

# Khởi tạo RL agent
rl_agent = RLPolicyAgent(input_dim, output_dim)
# Trong quá trình training, ta sẽ gọi trực tiếp rl_agent.select_action(state)
env.scheduler = None  

training_rewards = []

for episode in range(1, num_training_episodes + 1):
    state = env.reset()
    episode_reward = 0
    done = False
    while not done:
        action = rl_agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        rl_agent.rewards.append(reward)
        episode_reward += reward
        state = next_state
    rl_agent.finish_episode()
    training_rewards.append(episode_reward)
    print(f"Training Episode {episode}, Reward: {episode_reward:.2f}")
    
    # Lưu weight sau mỗi 32 episode
    if episode % save_interval == 0:
        weight_path = f"weights/rl_policy_episode_{episode}.pth"
        torch.save(rl_agent.policy_net.state_dict(), weight_path)
        print(f"Saved weights to {weight_path}")

# Vẽ learning curve
plt.figure(figsize=(10,6))
plt.plot(range(1, num_training_episodes + 1), training_rewards, label="Training Reward", color='b')
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Learning Curve for RL Scheduler")
plt.legend()
plt.grid(True)
plt.savefig("data/learning_curve.png")
plt.show()
