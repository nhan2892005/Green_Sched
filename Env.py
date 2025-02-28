import gym
import numpy as np
from Cluster import Cluster
from EnergySystem import EnergySystem
from Job import Job

JOB_QUEUE_SIZE = 64
NUM_JOB = 256
TIMESTEP = 300

class HPCEnv(gym.Env):
    def __init__(self):
        super(HPCEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(JOB_QUEUE_SIZE)
        # Observation gồm:
        # - Thông tin job queue (mỗi job 4 đặc trưng)
        # - Thông tin running jobs (mỗi job 4 đặc trưng)
        # - Thông tin hệ thống (7 đặc trưng)
        # - Thông tin năng lượng (5 đặc trưng)
        # Tổng cộng: 64*4 + 64*4 + 7 + 5 = 524 features
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(524,), dtype=np.float32)
        
        self.simulation_length = TIMESTEP
        self.cluster = Cluster(total_resources={'cpu': 200, 'ram': 256})
        self.energy_system = EnergySystem(solar_capacity=120, wind_capacity=600, battery_capacity=100)
        self.time = 0
        self.job_queue = []
        self.back_log = []
        self.generate_jobs()
        
        self.last_clean_energy = 0.0
        self.last_solar = 0.0
        self.last_wind = 0.0
        self.last_cluster_consumption = 0.0
        self.last_non_clean_energy = 0.0
        
        self.log_data = []

    def generate_jobs(self):
        for i in range(NUM_JOB):
            cpu_required = np.random.randint(5, 21) + np.random.rand()
            ram_required = np.random.randint(4, 17) + np.random.rand()
            duration = np.random.rand() * 7 + 0.5  # đảm bảo duration không quá nhỏ
            submit_time = 0  # tất cả job có thể submit ngay từ đầu
            job = Job(
                job_id=i,
                required_resources={'cpu': cpu_required, 'ram': ram_required},
                duration=duration,
                submit_time=submit_time
            )
            if i < JOB_QUEUE_SIZE:
                self.job_queue.append(job)
            else:
                self.back_log.append(job)
                
    def _get_observation(self):
        MAX_QUEUE_JOBS = JOB_QUEUE_SIZE  # 64
        queue_features = np.zeros((MAX_QUEUE_JOBS, 4))  # mỗi job: [submit_time, duration, waiting_time, required_cpu]
        for i, job in enumerate(self.job_queue[:MAX_QUEUE_JOBS]):
            queue_features[i] = [
                job.submit_time / self.simulation_length,
                job.duration / 10.0,  # giả sử duration max khoảng 10
                (self.time - job.submit_time) / self.simulation_length,
                job.required_resources['cpu'] / self.cluster.total_resources['cpu']
            ]
        
        MAX_RUNNING_JOBS = JOB_QUEUE_SIZE  # 64
        running_features = np.zeros((MAX_RUNNING_JOBS, 4))  # mỗi job: [start_time, progress, energy_used, required_cpu]
        for i, job in enumerate(self.cluster.running_jobs[:MAX_RUNNING_JOBS]):
            running_time = self.time - job.start_time
            progress = np.clip(running_time / job.duration, 0, 1)
            energy_used = job.energy_requirement * progress
            running_features[i] = [
                job.start_time / self.simulation_length,
                progress,
                energy_used / (job.energy_requirement + 1e-5),
                job.required_resources['cpu'] / self.cluster.total_resources['cpu']
            ]
        
        system_info = np.array([
            self.cluster.get_cpu_usage_ratio(),
            self.cluster.get_ram_usage_ratio(),
            len(self.back_log) / (NUM_JOB - JOB_QUEUE_SIZE + 1e-5),
            self.last_cluster_consumption / 100.0,  # chuẩn hóa
            self.time / self.simulation_length,
            len(self.job_queue) / JOB_QUEUE_SIZE,
            len(self.cluster.running_jobs) / MAX_RUNNING_JOBS
        ])
        
        if self.last_cluster_consumption > 0:
            brown_ratio = self.last_non_clean_energy / self.last_cluster_consumption
        else:
            brown_ratio = 0
        
        energy_info = np.array([
            self.last_clean_energy / (self.energy_system.solar_capacity + self.energy_system.wind_capacity),
            self.last_solar / (self.energy_system.solar_capacity + 1e-5),
            self.last_wind / (self.energy_system.wind_capacity + 1e-5),
            self.energy_system.battery_level / self.energy_system.battery_capacity,
            brown_ratio
        ])
        
        obs = np.concatenate([
            queue_features.flatten(),
            running_features.flatten(),
            system_info,
            energy_info
        ])
        return obs.astype(np.float32)
    
    def step(self, action):
        reward = 0
        done = False

        # Cập nhật hệ thống năng lượng
        clean_energy, solar_gen, wind_gen = self.energy_system.update(self.time)
        self.last_clean_energy = clean_energy
        self.last_solar = solar_gen
        self.last_wind = wind_gen

        # Cập nhật trạng thái cụm: giải phóng các job đã hoàn thành
        finished_jobs = self.cluster.update(self.time)
        self.last_cluster_consumption = sum(job.energy_requirement for job in self.cluster.running_jobs)

        # Nếu clean energy không đủ (dưới 20% tiêu thụ của cluster), dùng thêm từ pin
        if clean_energy < self.last_cluster_consumption * 0.2:
            clean_energy += self.energy_system.use_battery()

        non_clean_used = max(0, self.last_cluster_consumption - clean_energy)
        self.last_non_clean_energy = non_clean_used

        # Thực hiện hành động: chọn job từ hàng đợi theo index action
        if self.job_queue and action < len(self.job_queue):
            job = self.job_queue[action]
            if job.submit_time <= self.time and self.cluster.can_schedule(job):
                self.cluster.schedule_job(job, self.time)
                self.job_queue.pop(action)
                if self.back_log:
                    self.job_queue.append(self.back_log.pop(0))
            else:
                reward -= 1  # phạt nếu chọn job không hợp lệ

        # Tính toán average job slowdown cho các job đã hoàn thành
        slowdown_list = []
        threshold = 1.0  # tránh chia cho số quá nhỏ
        for job in finished_jobs:
            wait_time = job.start_time - job.submit_time
            run_time = job.finish_time - job.start_time
            slowdown = (wait_time + run_time) / max(run_time, threshold)
            slowdown_list.append(slowdown)
        avg_slowdown = np.mean(slowdown_list) if slowdown_list else 0

        # Tính % brown energy usage
        if self.last_cluster_consumption > 0:
            brown_pct = self.last_non_clean_energy / self.last_cluster_consumption
        else:
            brown_pct = 0

        # Sử dụng các hệ số trọng số để điều chỉnh ảnh hưởng của mỗi thành phần
        w1 = 1.0
        w2 = 10.0

        # Reward: hệ thống được thưởng khi slowdown và % brown energy thấp
        reward = - (w1 * avg_slowdown + w2 * brown_pct)

        self.time += 1
        if self.time >= self.simulation_length:
            done = True

        observation = self._get_observation()

        # Lưu log dữ liệu
        log_entry = {
            'time': self.time,
            'free_cpu_ratio': 1 - self.cluster.get_cpu_usage_ratio(),
            'job_queue_ratio': len(self.job_queue) / JOB_QUEUE_SIZE,
            'battery_ratio': self.energy_system.battery_level / self.energy_system.battery_capacity,
            'running_job_ratio': len(self.cluster.running_jobs) / JOB_QUEUE_SIZE,
            'time_norm': self.time / self.simulation_length,
            'cpu_usage_ratio': self.cluster.get_cpu_usage_ratio(),
            'clean_energy_ratio': min(clean_energy / self.last_cluster_consumption, 1.0) if self.last_cluster_consumption > 0 else 1.0,
            'solar_generation': solar_gen,
            'wind_generation': wind_gen,
            'clean_energy_generation': clean_energy,
            'non_clean_energy_used': non_clean_used,
            'cluster_consumption': self.last_cluster_consumption,
            'reward': reward,
            'battery_level': self.energy_system.battery_level,
            'ram_usage_ratio': self.cluster.get_ram_usage_ratio(),
            'brown_energy_ratio': brown_pct,
            'completed_jobs': len(finished_jobs),
            'avg_slowdown': avg_slowdown
        }
        self.log_data.append(log_entry)

        return observation, reward, done, {}

    def reset(self):
        self.time = 0
        self.cluster = Cluster(total_resources={'cpu': 200, 'ram': 256})
        self.energy_system = EnergySystem(solar_capacity=120, wind_capacity=600, battery_capacity=100)
        self.job_queue = []
        self.back_log = []
        self.generate_jobs()
        self.last_clean_energy = 0.0
        self.last_solar = 0.0
        self.last_wind = 0.0
        self.last_cluster_consumption = 0.0
        self.last_non_clean_energy = 0.0
        self.log_data = []
        return self._get_observation()

    def render(self, mode='human'):
        print(f"Time: {self.time}")
        print(f"Cluster CPU: {self.cluster.available_resources['cpu']} / {self.cluster.total_resources['cpu']}")
        print(f"Cluster RAM: {self.cluster.available_resources.get('ram',0)} / {self.cluster.total_resources.get('ram',0)}")
        print(f"CPU Usage Ratio: {self.cluster.get_cpu_usage_ratio():.2f}")
        print(f"Battery Level: {self.energy_system.battery_level:.2f} / {self.energy_system.battery_capacity}")
        print(f"Jobs in Queue: {len(self.job_queue)}")
        print(f"Solar: {self.last_solar:.2f}, Wind: {self.last_wind:.2f}, Clean Energy: {self.last_clean_energy:.2f}")
        print(f"Cluster Consumption: {self.last_cluster_consumption:.2f}")
        print(f"Non-Clean Energy Used: {self.last_non_clean_energy:.2f}")
        if self.last_cluster_consumption > 0:
            print(f"Clean Energy Ratio: {min(self.last_clean_energy / self.last_cluster_consumption, 1.0):.2f}")
        else:
            print("Clean Energy Ratio: 1.0")

    def get_action(self):
        return self.scheduler.schedule(self.job_queue, self.cluster, self.energy_system, self.time)
