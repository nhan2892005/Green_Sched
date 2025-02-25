BASE_ENERGY_PER_CPU = 0.8  # Energy consumed per unit of CPU per timestep
BASE_ENERGY_PER_RAM = 0.3  # Energy consumed per unit of RAM per timestep

'''
Job class: To represent a job in the simulation environment.
    * job_id: The unique identifier of the job.
    * required_resources: A dictionary that can contain keys: 'cpu', 'ram'.
    * duration: The number of timesteps the job runs.
    * value: The value of the job.
    * energy_requirement: The energy required to run the job.
    * start_time: The timestep when the job starts.
    * finish_time: The timestep when the job finishes.
'''
class Job:
    def __init__(self, job_id, required_resources, duration, submit_time):
        ''''''
        self.job_id = job_id
        self.required_resources = required_resources 
        self.duration = duration  
        self.submit_time = submit_time
        self.energy_requirement = (
            self.required_resources.get("cpu", 0) * BASE_ENERGY_PER_CPU +
            self.required_resources.get("ram", 0) * BASE_ENERGY_PER_RAM
        ) * self.duration
        self.start_time = None
        self.finish_time = None
        self.resource_start_unit = [] # [{'cpu': 3, 'ram': 4}]
        self.num_green_energy_used = 0
        self.num_brown_energy_used = 0
        
    def __repr__(self):
        return f"Job(id={self.job_id}, duration={self.duration}, submit_time={self.submit_time:.2f}, energy={self.energy_requirement:.2f})"
