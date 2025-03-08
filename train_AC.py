import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

from Env import HPCEnv

from RLBrain_ActorCritic import ActorCriticAgent

# Đặt seed cho tính nhất quán
np.random.seed(42)
torch.manual_seed(42)

# -------------------------------
# Huấn luyện bằng Actor-Critic với CNN
# -------------------------------
num_training_episodes = 1024
save_interval = 32

os.makedirs("weights", exist_ok=True)
os.makedirs("data", exist_ok=True)

env = HPCEnv(train=True)
input_dim = env.observation_space.shape[0]  # ví dụ 524
output_dim = env.action_space.n             # ví dụ 64

agent = ActorCriticAgent(input_dim, output_dim, hidden_dim=256, lr=1e-3)

training_rewards = []

for episode in range(1, num_training_episodes + 1):
    state = env.reset()
    episode_reward = 0
    done = False
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.rewards.append(reward)
        episode_reward += reward
        state = next_state
    agent.finish_episode()
    training_rewards.append(episode_reward)
    print(f"Episode {episode}, Reward: {episode_reward:.2f}")
    
    if episode % save_interval == 0:
        weight_path = f"weights/actor_critic/actor_critic_episode_{episode}.pth"
        torch.save({
            'actor_state_dict': agent.actor.state_dict(),
            'critic_state_dict': agent.critic.state_dict()
        }, weight_path)
        print(f"Saved weights to {weight_path}")

plt.figure(figsize=(10,6))
plt.plot(range(1, num_training_episodes + 1), training_rewards, label="Training Reward", color='b')
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Learning Curve for CNN-based Actor-Critic Scheduler")
plt.legend()
plt.grid(True)
plt.savefig("data/learning_curve_cnn.png")
plt.show()
