'''
Cluster class:
* total_resources: A dictionary that contains the total resources available in the cluster.
* available_resources: A dictionary that contains the available resources in the cluster.
* running_jobs: A list of jobs that are currently running in the cluster.
'''
class Cluster:
    def __init__(self, total_resources):
        # Ex: {'cpu': 100, 'ram': 256}
        self.total_resources = total_resources.copy()
        self.available_resources = total_resources.copy()
        self.running_jobs = []
        
    def can_schedule(self, job):
        for resource, amount in job.required_resources.items():
            if self.available_resources.get(resource, 0) < amount:
                return False
        return True
    
    def schedule_job(self, job, current_time):
        if self.can_schedule(job):
            for resource, amount in job.required_resources.items():
                self.available_resources[resource] -= amount
            job.start_time = current_time
            self.running_jobs.append(job)
            return True
        return False
    
    def update(self, current_time):
        finished = []
        for job in self.running_jobs:
            if job.start_time + job.duration <= current_time:
                for resource, amount in job.required_resources.items():
                    self.available_resources[resource] += amount
                job.finish_time = current_time
                finished.append(job)
        for job in finished:
            self.running_jobs.remove(job)
        return finished

    def get_cpu_usage_ratio(self):
        total_cpu = self.total_resources.get('cpu', 1)
        used_cpu = total_cpu - self.available_resources.get('cpu', 0)
        return used_cpu / total_cpu

    def get_ram_usage_ratio(self):
        total_ram = self.total_resources.get('ram', 1)
        used_ram = total_ram - self.available_resources.get('ram', 0)
        return used_ram / total_ram