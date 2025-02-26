from Env import HPCEnv, JOB_QUEUE_SIZE
from RLBrain import RLPolicyAgent
import torch

def train_agent(num_episodes=500):
    env = HPCEnv()
    input_dim = env.observation_space.shape[0]  # 524 features
    output_dim = JOB_QUEUE_SIZE  # 64 actions
    agent = RLPolicyAgent(input_dim, output_dim)
    
    for episode in range(num_episodes):
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
        print(f"Episode {episode+1} Reward: {episode_reward:.2f}")
    return agent

def test_agent(agent, num_episodes=10):
    env = HPCEnv()
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        while not done:
            state_tensor = torch.from_numpy(state).float()
            with torch.no_grad():
                action_probs = agent.policy_net(state_tensor)
            # Chọn hành động theo argmax
            action = torch.argmax(action_probs).item()
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
        print(f"Test Episode {episode+1} Reward: {episode_reward:.2f}")

# ---------------------------
# Chạy huấn luyện và test
# ---------------------------

if __name__ == "__main__":
    trained_agent = train_agent(num_episodes=300)
    print("\nTesting agent after training:")
    test_agent(trained_agent, num_episodes=5)