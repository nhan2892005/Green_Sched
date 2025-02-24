import csv
import matplotlib.pyplot as plt
from Env import HPCEnv

# Chạy simulation, ghi log ra CSV và vẽ biểu đồ.
if __name__ == "__main__":
    env = HPCEnv()
    obs = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        # Ở đây sử dụng hành động ngẫu nhiên; sau này có thể thay bằng hành động của mô hình RL.
        action = env.get_action()
        obs, reward, done, info = env.step(action)
        total_reward += reward
        env.render()
    
    print("Total Reward:", total_reward)
    
    # Ghi log dữ liệu vào file CSV.
    csv_file = "hpc_simulation_log.csv"
    fieldnames = ['time', 'free_cpu_ratio', 'job_queue_ratio', 'battery_ratio', 
                  'running_job_ratio', 'time_norm', 'cpu_usage_ratio', 
                  'clean_energy_ratio', 'solar_generation', 'wind_generation',
                  'clean_energy_generation', 'non_clean_energy_used',
                  'cluster_consumption', 'reward', 'battery_level', 'ram_usage_ratio']
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in env.log_data:
            writer.writerow(entry)
    print(f"Log dữ liệu đã được ghi ra file: {csv_file}")
    
    # Vẽ biểu đồ của các năng lượng và quá trình sử dụng tài nguyên theo thời gian.
    times = [entry['time'] for entry in env.log_data]
    solar_levels = [entry['solar_generation'] for entry in env.log_data]
    wind_levels = [entry['wind_generation'] for entry in env.log_data]
    clean_energy_levels = [entry['clean_energy_generation'] for entry in env.log_data]
    non_clean_levels = [entry['non_clean_energy_used'] for entry in env.log_data]
    cluster_consumptions = [entry['cluster_consumption'] for entry in env.log_data]
    cpu_usages = [entry['cpu_usage_ratio'] for entry in env.log_data]
    ram_usages = [entry['ram_usage_ratio'] for entry in env.log_data]
    
    plt.figure(figsize=(14, 10))
    
    # Biểu đồ năng lượng sạch: năng lượng mặt trời, gió và tổng clean energy.
    plt.subplot(3, 1, 1)
    plt.plot(times, solar_levels, marker='o', label='Solar Generation')
    plt.plot(times, wind_levels, marker='s', label='Wind Generation')
    plt.plot(times, clean_energy_levels, marker='^', label='Total Clean Energy')
    plt.ylabel("Clean Energy (units)")
    plt.title("Clean energy variability")
    plt.legend()
    plt.grid(True)
    
    # Biểu đồ năng lượng không sạch sử dụng và tiêu thụ của cụm.
    plt.subplot(3, 1, 2)
    plt.plot(times, non_clean_levels, marker='o', color='red', label='Non-Clean Energy Used')
    plt.plot(times, cluster_consumptions, marker='s', color='purple', label='Cluster Consumption')
    plt.ylabel("Energy Consumption (units)")
    plt.title("Non-Clean Energy Usage and Cluster Consumption")
    plt.legend()
    plt.grid(True)
    
    # Biểu đồ quá trình sử dụng resource: CPU usage và RAM usage.
    plt.subplot(3, 1, 3)
    plt.plot(times, cpu_usages, marker='o', label='CPU Usage Ratio')
    plt.plot(times, ram_usages, marker='s', label='RAM Usage Ratio')
    plt.xlabel("Time (timestep)")
    plt.ylabel("Usage Ratio")
    plt.title("Resource Usage Process (CPU & RAM)")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
