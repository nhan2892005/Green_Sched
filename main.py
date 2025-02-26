import csv
import matplotlib.pyplot as plt
import numpy as np
from Env import HPCEnv
from Scheduler import FCFSScheduler, SJFScheduler, RoundRobinScheduler, RLScheduler

# Chạy simulation, ghi log ra CSV và vẽ biểu đồ để so sánh các scheduler
if __name__ == "__main__":
    # Chạy với nhiều scheduler khác nhau để so sánh
    schedulers = {
        "FCFS": FCFSScheduler(),
        "SJF": SJFScheduler(),
        "RoundRobin": RoundRobinScheduler(),
        #"RL": RLScheduler(model_path="rl_scheduler_model")  # Load pre-trained model
    }
    
    # Tắt chế độ training khi đánh giá
    if "RL" in schedulers:
        schedulers["RL"].training_mode = False
    
    results = {}
    all_metrics = {}
    
    for scheduler_name, scheduler in schedulers.items():
        print(f"Running simulation with {scheduler_name} scheduler...")
        
        env = HPCEnv()
        env.scheduler = scheduler  # Gán scheduler cho môi trường
        obs = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = env.get_action()  # Action từ scheduler
            obs, reward, done, info = env.step(action)
            total_reward += reward
        
        print(f"{scheduler_name} - Total Reward: {total_reward}")
        
        # Lưu kết quả để so sánh
        results[scheduler_name] = {
            "env": env,
            "total_reward": total_reward,
            "log_data": env.log_data
        }
        
        # Trích xuất metrics cho việc so sánh
        if env.log_data:
            # Brown energy ratio trung bình
            brown_ratios = [entry['brown_energy_ratio'] for entry in env.log_data]
            # Số job hoàn thành
            completed = env.log_data[-1]['completed_jobs'] if env.log_data else 0
            # Slowdown trung bình
            slowdowns = [entry['avg_slowdown'] for entry in env.log_data if entry['avg_slowdown'] > 0]
            avg_slowdown = np.mean(slowdowns) if slowdowns else 0
            # Resource utilization trung bình
            cpu_usage = np.mean([entry['cpu_usage_ratio'] for entry in env.log_data])
            
            all_metrics[scheduler_name] = {
                'brown_energy_ratio': np.mean(brown_ratios),
                'completed_jobs': completed,
                'avg_slowdown': avg_slowdown,
                'cpu_usage': cpu_usage,
                'total_reward': total_reward
            }
        
        # Ghi log dữ liệu vào file CSV
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

    # Vẽ biểu đồ so sánh các metrics
    metrics = ['brown_energy_ratio', 'completed_jobs', 'avg_slowdown', 'cpu_usage', 'total_reward']
    scheduler_names = list(all_metrics.keys())
    
    plt.figure(figsize=(16, 12))
    
    for i, metric in enumerate(metrics):
        plt.subplot(3, 2, i+1)
        values = [all_metrics[s][metric] for s in scheduler_names]
        bars = plt.bar(scheduler_names, values)
        
        # Add values on top of bars
        for bar, val in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, val + 0.05*max(values), 
                     f'{val:.2f}', ha='center')
        
        plt.title(f'Comparison of {metric.replace("_", " ").title()}')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('data/scheduler_comparison.png')
    
    # Vẽ biểu đồ cho tỷ lệ năng lượng không xanh theo thời gian
    plt.figure(figsize=(14, 8))
    
    for scheduler_name, result in results.items():
        times = [entry['time'] for entry in result['log_data']]
        brown_ratios = [entry['brown_energy_ratio'] for entry in result['log_data']]
        plt.plot(times, brown_ratios, marker='.', label=scheduler_name)
    
    plt.xlabel("Time (timestep)")
    plt.ylabel("Brown Energy Ratio")
    plt.title("Non-Green Energy Usage Over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig("data/brown_energy_comparison.png")