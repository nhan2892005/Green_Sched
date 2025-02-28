import os
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from Env import HPCEnv
from RLBrain_ActorCritic import ActorCriticAgent
from Scheduler import FCFSScheduler, SJFScheduler, RoundRobinScheduler

# Đặt seed cho tính nhất quán
np.random.seed(42)
torch.manual_seed(42)

# -------------------------------
# RL Scheduler Wrapper (Test Mode)
# -------------------------------
class RLTestScheduler:
    def __init__(self, agent):
        self.agent = agent
        self.training_mode = False
    def schedule(self, job_queue, cluster, energy_system, current_time, observation):
        # Ở chế độ test, chọn action theo argmax (greedy)
        state_tensor = torch.from_numpy(observation).float().unsqueeze(0)  # (1, input_dim)
        with torch.no_grad():
            probs = self.agent.actor(state_tensor)  # sử dụng actor của mô hình
        return torch.argmax(probs).item()

# Tạo thư mục nếu chưa có
os.makedirs("data", exist_ok=True)

# Khởi tạo môi trường và RL agent
env = HPCEnv()
input_dim = env.observation_space.shape[0]  # ví dụ: 524
output_dim = env.action_space.n             # ví dụ: 64

agent = ActorCriticAgent(input_dim, output_dim, hidden_dim=256, lr=1e-3)

# Nếu đã huấn luyện, load weight từ file (ví dụ sử dụng weight cuối cùng)
weight_file = "weights/actor_critic/actor_critic_episode_1024.pth"
if os.path.exists(weight_file):
    checkpoint = torch.load(weight_file)
    agent.actor.load_state_dict(checkpoint['actor_state_dict'])
    agent.critic.load_state_dict(checkpoint['critic_state_dict'])
    print(f"Loaded weights from {weight_file}")
else:
    print("Weight file not found, using randomly initialized weights.")

# Khởi tạo RL Test Scheduler
rl_test_scheduler = RLTestScheduler(agent)

# Khai báo các scheduler để so sánh
schedulers = {
    "FCFS": FCFSScheduler(),
    "SJF": SJFScheduler(),
    "RoundRobin": RoundRobinScheduler(),
    "RL": rl_test_scheduler
}

results = {}     # Lưu kết quả simulation cho từng scheduler
all_metrics = {} # Lưu các metric so sánh

for scheduler_name, scheduler in schedulers.items():
    print(f"\nRunning simulation with {scheduler_name} scheduler...")
    # Tạo mới môi trường cho từng simulation (có thể dùng deepcopy nếu cần đảm bảo cùng trạng thái ban đầu)
    env = HPCEnv()
    env.scheduler = scheduler  # Gán scheduler vào môi trường
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        if scheduler_name == "RL":
            # Với RL, sử dụng hàm schedule của RLTestScheduler (cần truyền observation)
            action = scheduler.schedule(env.job_queue, env.cluster, env.energy_system, env.time, state)
        else:
            action = scheduler.schedule(env.job_queue, env.cluster, env.energy_system, env.time)
        state, reward, done, _ = env.step(action)
        total_reward += reward
    print(f"{scheduler_name} - Total Reward: {total_reward:.2f}")
    results[scheduler_name] = {
        "env": env,
        "total_reward": total_reward,
        "log_data": env.log_data
    }
    
    # Trích xuất các metric từ log_data
    if env.log_data:
        brown_ratios = [entry['brown_energy_ratio'] for entry in env.log_data]
        completed = env.log_data[-1]['completed_jobs'] if env.log_data else 0
        slowdowns = [entry['avg_slowdown'] for entry in env.log_data if entry['avg_slowdown'] > 0]
        avg_slowdown = np.mean(slowdowns) if slowdowns else 0
        cpu_usage = np.mean([entry['cpu_usage_ratio'] for entry in env.log_data])
        all_metrics[scheduler_name] = {
            'brown_energy_ratio': np.mean(brown_ratios),
            'completed_jobs': completed,
            'avg_slowdown': avg_slowdown,
            'cpu_usage': cpu_usage,
            'total_reward': total_reward
        }
    
    # Ghi log dữ liệu của simulation vào file CSV
    csv_file = f"data/hpc_simulation_log_{scheduler_name}.csv"
    fieldnames = ['time', 'free_cpu_ratio', 'job_queue_ratio', 'battery_ratio', 
                  'running_job_ratio', 'time_norm', 'cpu_usage_ratio', 
                  'clean_energy_ratio', 'solar_generation', 'wind_generation',
                  'clean_energy_generation', 'non_clean_energy_used',
                  'cluster_consumption', 'reward', 'battery_level', 'ram_usage_ratio',
                  'brown_energy_ratio', 'completed_jobs', 'avg_slowdown']
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in env.log_data:
            writer.writerow(entry)
    print(f"Log data written to: {csv_file}")

# -------------------------------
# Vẽ biểu đồ so sánh các metric giữa các Scheduler
# -------------------------------
metrics = ['brown_energy_ratio', 'completed_jobs', 'avg_slowdown', 'cpu_usage', 'total_reward']
scheduler_names = list(all_metrics.keys())

plt.figure(figsize=(16, 12))
for i, metric in enumerate(metrics):
    plt.subplot(3, 2, i+1)
    values = [all_metrics[s][metric] for s in scheduler_names]
    bars = plt.bar(scheduler_names, values, color=['#4daf4a','#377eb8','#ff7f00','#e41a1c'])
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, val + 0.05*max(values), f'{val:.2f}', ha='center', fontsize=10)
    plt.title(f'Comparison of {metric.replace("_", " ").title()}', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('data/scheduler_comparison_actorcritic.png')
plt.show()

# Biểu đồ đường cho tỷ lệ năng lượng không xanh theo thời gian
plt.figure(figsize=(14, 8))
for scheduler_name, result in results.items():
    times = [entry['time'] for entry in result['log_data']]
    brown_ratios = [entry['brown_energy_ratio'] for entry in result['log_data']]
    plt.plot(times, brown_ratios, marker='.', label=scheduler_name)
plt.xlabel("Time (timestep)", fontsize=12)
plt.ylabel("Brown Energy Ratio", fontsize=12)
plt.title("Non-Green Energy Usage Over Time", fontsize=14)
plt.legend()
plt.grid(True)
plt.savefig("data/brown_energy_comparison_actorcritic.png")
plt.show()

# Biểu đồ boxplot cho các metric chọn lọc (Step Reward, CPU Usage, Brown Energy Ratio, Average Slowdown)
metrics_to_plot = {
    "Brown Energy Ratio": "brown_energy_ratio",
    "Average Slowdown": "avg_slowdown",
    "CPU Usage Ratio": "cpu_usage_ratio",
    "Step Reward": "reward"
}
num_metrics = len(metrics_to_plot)
fig, axes = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 6), sharey=False)
for i, (metric_label, metric_key) in enumerate(metrics_to_plot.items()):
    data = []
    labels = []
    for scheduler_name, result in results.items():
        metric_values = [entry[metric_key] for entry in result['log_data'] if metric_key in entry]
        data.append(metric_values)
        labels.append(scheduler_name)
    axes[i].boxplot(data, labels=labels, showmeans=True)
    axes[i].set_title(metric_label, fontsize=12)
    axes[i].set_xlabel("Scheduler", fontsize=10)
    axes[i].grid(axis="y", linestyle="--", alpha=0.7)
    if metric_label == "Step Reward":
        axes[i].set_ylabel("Reward", fontsize=10)
    elif metric_label == "CPU Usage Ratio":
        axes[i].set_ylabel("CPU Usage Ratio", fontsize=10)
    elif metric_label == "Brown Energy Ratio":
        axes[i].set_ylabel("Brown Energy Ratio", fontsize=10)
    elif metric_label == "Average Slowdown":
        axes[i].set_ylabel("Slowdown", fontsize=10)
plt.tight_layout()
plt.savefig("data/boxplot_analysis_actorcritic.png")
plt.show()
