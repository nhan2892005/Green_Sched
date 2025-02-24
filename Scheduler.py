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