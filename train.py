import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from Env import HPCEnv 
from Scheduler import RLScheduler, PolicyGradient

# Thiết lập seed cho tính tái tạo
np.random.seed(1)
tf.random.set_seed(1)

def train_rl_scheduler(n_episodes=1000, save_path="rl_scheduler_model"):
    """
    Huấn luyện RL Scheduler
    """
    env = HPCEnv()
    rl_scheduler = RLScheduler()
    env.scheduler = rl_scheduler
    
    # Theo dõi reward qua các episodes
    episode_rewards = []
    brown_energy_ratios = []
    job_completion_rates = []
    
    for episode in range(n_episodes):
        # Reset môi trường
        observation = env.reset()
        done = False
        episode_reward = 0
        
        # Lưu trữ transitions cho batch training
        all_observations = []
        all_actions = []
        all_rewards = []
        
        while not done:
            # Reshape observation để phù hợp với model input
            obs_reshaped = np.reshape(observation, (1, -1))
            all_observations.append(obs_reshaped[0])
            
            # Chọn action dựa trên observation
            action = rl_scheduler.schedule(env.job_queue, env.cluster, env.energy_system, env.time)
            all_actions.append(action)
            
            # Thực hiện action và nhận state, reward mới
            observation, reward, done, info = env.step(action)
            all_rewards.append(reward)
            episode_reward += reward
            
        # Convert to numpy arrays for batch learning
        all_observations = np.array(all_observations)
        all_actions = np.array(all_actions)
        
        # Discount và normalize rewards
        discounted_rewards = np.zeros_like(all_rewards, dtype=np.float32)
        running_add = 0
        for t in reversed(range(len(all_rewards))):
            running_add = running_add * rl_scheduler.rl_agent.gamma + all_rewards[t]
            discounted_rewards[t] = running_add
            
        # Normalize rewards
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards) + 1e-8  # Tránh chia cho 0
        
        # Train model
        loss = rl_scheduler.rl_agent.learn(all_observations, all_actions, discounted_rewards)
        
        # Lưu metrics
        episode_rewards.append(episode_reward)
        
        # Tính brown energy ratio trung bình
        if env.log_data:
            avg_brown_ratio = np.mean([entry['brown_energy_ratio'] for entry in env.log_data])
            brown_energy_ratios.append(avg_brown_ratio)
            
            # Tính tỷ lệ job hoàn thành
            completed_jobs = env.log_data[-1]['completed_jobs'] if env.log_data else 0
            job_completion_rates.append(completed_jobs / 64)  # Giả sử tổng số job là 64
        
        # Log progress
        if episode % 10 == 0:
            print(f"Episode {episode}/{n_episodes}, Reward: {episode_reward:.2f}, Loss: {loss:.4f}")
            if env.log_data:
                print(f"  Brown Energy Ratio: {avg_brown_ratio:.2f}, Completed Jobs: {completed_jobs}")
    
    # Lưu model đã train
    rl_scheduler.rl_agent.save_data(save_path)
    print(f"Model saved to {save_path}")
    
    # Vẽ biểu đồ learning curve
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    plt.subplot(1, 3, 2)
    plt.plot(brown_energy_ratios)
    plt.title('Brown Energy Usage')
    plt.xlabel('Episode')
    plt.ylabel('Brown Energy Ratio')
    
    plt.subplot(1, 3, 3)
    plt.plot(job_completion_rates)
    plt.title('Job Completion Rate')
    plt.xlabel('Episode')
    plt.ylabel('Completion Rate')
    
    plt.tight_layout()
    plt.savefig('rl_training_curves.png')
    
    return rl_scheduler

if __name__ == "__main__":
    train_rl_scheduler(n_episodes=500, save_path="rl_scheduler_model")