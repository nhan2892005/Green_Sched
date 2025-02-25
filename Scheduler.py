class Scheduler:
    def schedule(self, jobs, cluster, energy_system, current_time):
        raise NotImplementedError("Phải triển khai phương thức schedule trong lớp con.")

class FCFSScheduler(Scheduler):
    def schedule(self, jobs, cluster, energy_system, current_time):
        sorted_jobs_with_index = sorted(enumerate(jobs), key=lambda x: x[1].submit_time)
        for index, job in sorted_jobs_with_index:
            if cluster.can_schedule(job):
                return index
        return 0
    
class SJFScheduler(Scheduler):
    """
    Shortest Job First: Ưu tiên job có thời gian chạy ngắn nhất
    """
    def schedule(self, jobs, cluster, energy_system, current_time):
        if not jobs:
            return 0
            
        # Sắp xếp jobs theo thời gian chạy (duration) từ ngắn đến dài
        job_durations = [(idx, job.duration) for idx, job in enumerate(jobs) 
                        if job.submit_time <= current_time and cluster.can_schedule(job)]
        
        if not job_durations:
            return 0  # Không có job phù hợp
            
        # Chọn job có duration ngắn nhất
        job_durations.sort(key=lambda x: x[1])
        return job_durations[0][0]


class RoundRobinScheduler(Scheduler):
    """
    Round Robin: Chọn job theo vòng tròn
    """
    def __init__(self):
        self.last_index = -1
        
    def schedule(self, jobs, cluster, energy_system, current_time):
        if not jobs:
            return 0
            
        # Tìm job kế tiếp có thể schedule
        n = len(jobs)
        for i in range(n):
            idx = (self.last_index + i + 1) % n
            job = jobs[idx]
            if job.submit_time <= current_time and cluster.can_schedule(job):
                self.last_index = idx
                return idx
                
        return 0  # Không có job phù hợp


class RLScheduler(Scheduler):
    """
    Reinforcement Learning-based Scheduler
    """
    def __init__(self, model_path=None):
        self.n_features = 308  # Kích thước input feature
        self.n_actions = 64    # Số lượng action (JOB_QUEUE_SIZE)
        self.rl_agent = PolicyGradient(
            n_actions=self.n_actions,
            n_features=self.n_features,
            learning_rate=0.01
        )
        if model_path:
            try:
                self.rl_agent.load_data(model_path)
                print(f"Loaded RL model from {model_path}")
            except:
                print(f"Could not load model from {model_path}, using new model")
        
        self.training_mode = True
        
    def schedule(self, jobs, cluster, energy_system, current_time):
        if not jobs:
            return 0
            
        # Lấy observation state
        # Giả định môi trường đã tạo observation đúng
        obs = np.zeros((1, self.n_features))  # Cần tạo observation ở đây
        
        # Nếu đang ở chế độ training, chọn action theo policy
        if self.training_mode:
            action = self.rl_agent.choose_action(obs)
        else:
            # Chọn action tốt nhất theo model đã học
            action_probs = self.rl_agent.model.predict(obs)[0]
            valid_actions = []
            for idx, job in enumerate(jobs):
                if job.submit_time <= current_time and cluster.can_schedule(job):
                    valid_actions.append((idx, action_probs[idx]))
            
            if valid_actions:
                action = max(valid_actions, key=lambda x: x[1])[0]
            else:
                action = 0
                
        return action