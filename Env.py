# sudo python3 -m pip install gym --break-system-packages
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
JOB_QUEUE_SIZE = 64
NUM_JOB = 128
TIMESTEP = 1500

class HPCEnv(gym.Env):
    def __init__(self):
        super(HPCEnv, self).__init__()
        
        self.action_space = gym.spaces.Discrete(JOB_QUEUE_SIZE)
        # Observation gồm 7 thông số: free_cpu_ratio, job_queue_ratio, battery_ratio, running_job_ratio, time_norm, cpu_usage_ratio, clean_energy_ratio.
        self.observation_space = gym.spaces.Box(
            low=0, 
            high=1, 
            shape=(308,),  # New observation space size
            dtype=np.float32
        )
        self.simulation_length = TIMESTEP
        self.cluster = Cluster(total_resources={'cpu': 200, 'ram': 256})
        self.energy_system = EnergySystem(solar_capacity=120, wind_capacity=600, battery_capacity=100)
        self.scheduler = FCFSScheduler()
        
        self.time = 0
        self.time_limit = 100
        self.job_queue = []
        self.back_log = []
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
        for i in range(NUM_JOB):
            cpu_required = np.random.randint(5, 21) + np.random.rand()
            ram_required = np.random.randint(4, 17) + np.random.rand()
            duration = np.random.rand() * 7
            submit_time = 0 # np.random.rand() * 90
            job = Job(
                job_id=i,
                required_resources={'cpu': cpu_required, 'ram': ram_required},
                duration=duration,
                submit_time=submit_time
            )
            if (i < JOB_QUEUE_SIZE):
                self.job_queue.append(job)
            else:
                self.back_log.append(job)
            
    def step(self, action):
        reward = 0
        done = False
        
        # Update energy status
        clean_energy, solar_gen, wind_gen = self.energy_system.update(self.time)
        self.last_clean_energy = clean_energy
        self.last_solar = solar_gen
        self.last_wind = wind_gen
        
        # Update cluster status, release finished jobs
        finished_jobs = self.cluster.update(self.time)
        # Calculate energy consumption of the cluster for this timestep
        self.last_cluster_consumption = sum(job.energy_requirement for job in self.cluster.running_jobs)
        
        # Use clean energy storage if the clean energy is less than 20% of the cluster consumption
        if (clean_energy < self.last_cluster_consumption * 0.2):
            clean_energy += self.energy_system.use_battery()
        
        # Calculate non-clean energy used
        non_clean_used = max(0, self.last_cluster_consumption - clean_energy)
        self.last_non_clean_energy = non_clean_used
        
        # Get the action
        if self.job_queue and action < len(self.job_queue):
            job = self.job_queue[action]
            if job.submit_time <= self.time and self.cluster.can_schedule(job):
                self.cluster.schedule_job(job, self.time)
                self.job_queue.pop(action)
                # if backlog is not empty, add the first job in the backlog to the job queue
                if self.back_log:
                    self.job_queue.append(self.back_log.pop(0))
            else:
                reward -= 1  # Penalty for selecting invalid job
        
        # Calculate rewards
        # 1. Energy efficiency reward
        if self.last_cluster_consumption > 0:
            clean_energy_ratio = min(clean_energy / self.last_cluster_consumption, 1.0)
            # Reward for using clean energy, penalty for brown energy
            energy_reward = clean_energy_ratio * 10 - non_clean_used * 5
        else:
            energy_reward = 0
        
        # 2. Job completion reward
        completion_reward = len(finished_jobs)
        
        # 3. Bounded slowdown penalty
        slowdown_penalty = 0
        for job in finished_jobs:
            wait_time = job.start_time - job.submit_time
            run_time = job.finish_time - job.start_time
            threshold = 1.0  # Minimum threshold to avoid division by very small numbers
            bounded_slowdown = (wait_time + run_time) / max(run_time, threshold)
            slowdown_penalty += bounded_slowdown * 0.8
        
        # 4. System utilization reward
        utilization_reward = self.cluster.get_cpu_usage_ratio() * 3
        
        # Combine all reward components
        reward = energy_reward + completion_reward - slowdown_penalty + utilization_reward
        
        self.time += 1
        if self.time >= self.simulation_length:
            done = True
        
        # Get obs info
        observation = self._get_observation()
        
        # Log thông tin
        log_entry = {
            'time': self.time,
            'free_cpu_ratio': 1 - self.cluster.get_cpu_usage_ratio(),
            'job_queue_ratio': len(self.job_queue) / JOB_QUEUE_SIZE,
            'battery_ratio': self.energy_system.battery_level / self.energy_system.battery_capacity,
            'running_job_ratio': len(self.cluster.running_jobs) / JOB_QUEUE_SIZE,
            'time_norm': self.time / self.simulation_length,
            'cpu_usage_ratio': self.cluster.get_cpu_usage_ratio(),
            'clean_energy_ratio': clean_energy_ratio if self.last_cluster_consumption > 0 else 1.0,
            'solar_generation': solar_gen,
            'wind_generation': wind_gen,
            'clean_energy_generation': clean_energy,
            'non_clean_energy_used': non_clean_used,
            'cluster_consumption': self.last_cluster_consumption,
            'reward': reward,
            'battery_level': self.energy_system.battery_level,
            'ram_usage_ratio': self.cluster.get_ram_usage_ratio(),
            'brown_energy_ratio': non_clean_used / self.last_cluster_consumption if self.last_cluster_consumption > 0 else 0,
            'completed_jobs': len(finished_jobs),
            'avg_slowdown': slowdown_penalty / len(finished_jobs) if finished_jobs else 0
        }
        self.log_data.append(log_entry)
        
        return observation, reward, done, {}
        
    def _get_observation(self):
        # 1. Job Queue Information (cho mỗi job trong queue)
        MAX_QUEUE_JOBS = 64  # Số lượng job tối đa trong queue
        queue_features = np.zeros((MAX_QUEUE_JOBS, 4))  # [submit_time, duration, waiting_time, required_cpu]
        
        for i, job in enumerate(self.job_queue[:MAX_QUEUE_JOBS]):
            queue_features[i] = [
                job.submit_time / self.simulation_length,  # Normalized submit time
                job.duration,  # Normalized duration (giả sử max duration = 10)
                (self.time - job.submit_time) / self.simulation_length,  # Normalized waiting time
                job.required_resources['cpu'] / self.cluster.total_resources['cpu']  # Normalized CPU requirement
            ]
        
        # 2. Running Jobs Information
        MAX_RUNNING_JOBS = 64  # Số lượng job đang chạy tối đa
        running_features = np.zeros((MAX_RUNNING_JOBS, 4))  # [start_time, running_time, energy_used, required_cpu]
        
        for i, job in enumerate(self.cluster.running_jobs[:MAX_RUNNING_JOBS]):
            running_time = self.time - job.start_time
            running_features[i] = [
                job.start_time / self.simulation_length,  # Normalized start time
                running_time / job.duration,  # Normalized running progress
                job.energy_requirement * (running_time / job.duration),  # Energy used so far
                job.required_resources['cpu'] / self.cluster.total_resources['cpu']  # Normalized CPU requirement
            ]
        
        # 3. System Information
        system_info = np.array([
            self.cluster.get_cpu_usage_ratio(),  # CPU usage ratio
            self.cluster.get_ram_usage_ratio(),  # RAM usage ratio
            len(self.back_log) / (NUM_JOB - JOB_QUEUE_SIZE),  # Normalized backlog size
            self.last_cluster_consumption / 100.0,  # Normalized cluster consumption
            self.time / self.simulation_length,  # Normalized current time
            len(self.job_queue) / MAX_QUEUE_JOBS,  # Normalized queue length
            len(self.cluster.running_jobs) / MAX_RUNNING_JOBS  # Normalized running jobs
        ])
        
        # 4. Energy System Information
        energy_info = np.array([
            self.last_clean_energy / (self.energy_system.solar_capacity + self.energy_system.wind_capacity),
            self.last_solar / self.energy_system.solar_capacity,
            self.last_wind / self.energy_system.wind_capacity,
            self.energy_system.battery_level / self.energy_system.battery_capacity,
            self.last_non_clean_energy / self.last_cluster_consumption if self.last_cluster_consumption > 0 else 0
        ])
        
        # Concatenate all features
        obs = np.concatenate([
            queue_features.flatten(),  # 64 * 4 = 256 features
            running_features.flatten(),  # 64 * 4 = 256 features
            system_info,  # 7 features
            energy_info  # 5 features
        ])
        
        return obs.astype(np.float32)
        
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