import gym
import gym.spaces
import numpy as np
from EnergySystem import EnergySystem
from Job import Job
from Cluster import Cluster
from Scheduler import FCFSScheduler

'''
An HPC environment simulation
* action_space: Discrete space with 2 actions: 0 - no scheduling, 1 - schedule a job from the queue.
* observation_space: Box space with 7 features: 
    - free_cpu_ratio
    - job_queue_ratio
    - battery_ratio, 
    - running_job_ratio
    - time_norm
    - cpu_usage_ratio
    - clean_energy_ratio
* simulation_length: Number of timesteps in the simulation.
* time: Current timestep in the simulation.
* job_queue: A list of jobs waiting to be scheduled.
* last_clean_energy: The amount of clean energy generated in the last timestep.
* last_solar: The amount of solar energy generated in the last timestep.
* last_wind: The amount of wind energy generated in the last timestep.
* last_cluster_consumption: The total energy consumed by the cluster in the last timestep.
* last_non_clean_energy: The amount of non-clean energy used in the last timestep.
* log_data: A list to store the simulation data for each timestep.
'''
class HPCEnv(gym.Env):
    def __init__(self):
        super(HPCEnv, self).__init__()
        # Định nghĩa không gian hành động: laf index cua job trong hàng đợi.
        self.action_space = gym.spaces.Discrete(64)
        # Observation gồm 7 thông số: free_cpu_ratio, job_queue_ratio, battery_ratio, running_job_ratio, time_norm, cpu_usage_ratio, clean_energy_ratio.
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(7,), dtype=np.float32)
        
        self.simulation_length = 100  # Số timestep của simulation
        # Cụm HPC có cả CPU và RAM.
        self.cluster = Cluster(total_resources={'cpu': 100, 'ram': 256})
        self.energy_system = EnergySystem(solar_capacity=50, wind_capacity=30, battery_capacity=100)
        self.scheduler = FCFSScheduler()
        
        self.time = 0
        self.job_queue = []
        self.generate_jobs()  # Tạo 64 job
        self.last_clean_energy = 0.0
        self.last_solar = 0.0
        self.last_wind = 0.0
        self.last_cluster_consumption = 0.0
        self.last_non_clean_energy = 0.0
        
        # Log dữ liệu simulation.
        self.log_data = []

    def get_action(self):
        return self.scheduler.schedule(self.job_queue, self.cluster, self.energy_system, self.time)
        
    def generate_jobs(self):
        for i in range(64):
            cpu_required = np.random.randint(5, 21) + np.random.rand()
            ram_required = np.random.randint(4, 17) + np.random.rand()
            duration = np.random.rand() * 7
            submit_time = np.random.rand() * 90
            job = Job(
                job_id=i,
                required_resources={'cpu': cpu_required, 'ram': ram_required},
                duration=duration,
                submit_time=submit_time
            )
            self.job_queue.append(job)
            
    def step(self, action):
        reward = 0
        done = False
        
        # Cập nhật hệ thống năng lượng.
        clean_energy, solar_gen, wind_gen = self.energy_system.update(self.time)
        self.last_clean_energy = clean_energy
        self.last_solar = solar_gen
        self.last_wind = wind_gen
        
        # Cập nhật trạng thái cụm: giải phóng các job đã hoàn thành.
        finished_jobs = self.cluster.update(self.time)
        self.last_cluster_consumption = sum(job.energy_requirement for job in self.cluster.running_jobs)
        
        # Nếu clean energy không đủ, phần thiếu sẽ được bù bởi nguồn điện không sạch.
        non_clean_used = max(0, self.last_cluster_consumption - clean_energy)
        self.last_non_clean_energy = non_clean_used
        
        # Nếu action == 1, lấy job từ hàng đợi để đặt lịch.
        # Get job base on action index
        if self.job_queue:
            job = self.job_queue.pop(action)
            if self.cluster.can_schedule(job) and job.submit_time <= self.time:
                self.cluster.schedule_job(job, self.time)
            else:
                self.job_queue.append(job)
                
        # Reward: tổng giá trị của các job hoàn thành trừ phạt khi sử dụng năng lượng không sạch.
        reward -= sum((job.finish_time - job.submit_time) for job in finished_jobs)
        reward -= non_clean_used
        
        self.time += 1
        if self.time >= self.simulation_length:
            done = True
        
        observation = self._get_observation()
        
        # Log thông tin của timestep hiện tại, bao gồm cả tỉ lệ sử dụng RAM.
        log_entry = {
            'time': self.time,
            'free_cpu_ratio': observation[0],
            'job_queue_ratio': observation[1],
            'battery_ratio': observation[2],
            'running_job_ratio': observation[3],
            'time_norm': observation[4],
            'cpu_usage_ratio': observation[5],
            'clean_energy_ratio': observation[6],
            'solar_generation': solar_gen,
            'wind_generation': wind_gen,
            'clean_energy_generation': clean_energy,
            'non_clean_energy_used': non_clean_used,
            'cluster_consumption': self.last_cluster_consumption,
            'reward': reward,
            'battery_level': self.energy_system.battery_level,
            'ram_usage_ratio': self.cluster.get_ram_usage_ratio()
        }
        self.log_data.append(log_entry)
        
        return observation, reward, done, {}
        
    def _get_observation(self):
        free_cpu_ratio = self.cluster.available_resources['cpu'] / self.cluster.total_resources['cpu']
        cpu_usage_ratio = 1 - free_cpu_ratio
        job_queue_ratio = len(self.job_queue) / 50.0  # Vì max job là 50
        battery_ratio = self.energy_system.battery_level / self.energy_system.battery_capacity
        running_job_ratio = len(self.cluster.running_jobs) / 10.0  # Giả sử tối đa 10 job chạy đồng thời
        time_norm = self.time / self.simulation_length
        if self.last_cluster_consumption > 0:
            clean_energy_ratio = min(self.last_clean_energy / self.last_cluster_consumption, 1.0)
        else:
            clean_energy_ratio = 1.0
        obs = np.array([
            free_cpu_ratio,
            job_queue_ratio,
            battery_ratio,
            running_job_ratio,
            time_norm,
            cpu_usage_ratio,
            clean_energy_ratio
        ], dtype=np.float32)
        return obs
        
    def reset(self):
        self.time = 0
        self.cluster = Cluster(total_resources={'cpu': 100, 'ram': 256})
        self.energy_system = EnergySystem(solar_capacity=50, wind_capacity=30, battery_capacity=100)
        self.job_queue = []
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
        print(f"Cluster Available CPU: {self.cluster.available_resources['cpu']} / {self.cluster.total_resources['cpu']}")
        print(f"Cluster Available RAM: {self.cluster.available_resources.get('ram',0)} / {self.cluster.total_resources.get('ram',0)}")
        print(f"CPU Usage Ratio: {self.cluster.get_cpu_usage_ratio():.2f}")
        print(f"RAM Usage Ratio: {self.cluster.get_ram_usage_ratio():.2f}")
        print(f"Battery Level: {self.energy_system.battery_level:.2f} / {self.energy_system.battery_capacity}")
        print(f"Jobs in Queue: {len(self.job_queue)}")
        print(f"Solar Generation: {self.last_solar:.2f}, Wind Generation: {self.last_wind:.2f}, Clean Energy: {self.last_clean_energy:.2f}")
        print(f"Cluster Consumption: {self.last_cluster_consumption:.2f}")
        print(f"Non-Clean Energy Used: {self.last_non_clean_energy:.2f}")
        if self.last_cluster_consumption > 0:
            clean_ratio = min(self.last_clean_energy / self.last_cluster_consumption, 1.0)
        else:
            clean_ratio = 1.0
        print(f"Clean Energy Ratio: {clean_ratio:.2f}")