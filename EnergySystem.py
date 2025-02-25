import numpy as np

'''
EnergySystem class:
* solar_capacity: maximum solar energy generation capacity.
* wind_capacity: maximum wind energy generation capacity.
* battery_capacity: clean energy storage capacity.
* battery_level: current battery level.
'''
class EnergySystem:
    def __init__(self, solar_capacity, wind_capacity, battery_capacity):
        self.solar_capacity = solar_capacity    # công suất tối đa của năng lượng mặt trời
        self.wind_capacity = wind_capacity      # công suất tối đa của năng lượng gió
        self.battery_capacity = battery_capacity  # dung lượng pin lưu trữ clean energy
        self.battery_level = battery_capacity      # bắt đầu đầy pin
        
    def update(self, time_step):
        # Giả sử mỗi timestep là 1 giờ, sử dụng modulo 24 để mô phỏng chu kỳ ngày đêm.
        hour = time_step % 24
        # Năng lượng mặt trời: hoạt động từ 6h đến 18h, peak tại 12h.
        solar_generation = self.solar_capacity * np.clip(np.sin(np.pi * (hour - 6) / 12), 0, None)
        # Năng lượng gió: biến động theo chu kỳ và nhiễu ngẫu nhiên.
        wind_generation = self.wind_capacity * (0.5 + 0.3 * np.sin(np.pi * hour / 24) + np.random.uniform(-1, 1))
        wind_generation = max(wind_generation, 0)
        clean_energy_generation = solar_generation + wind_generation
        
        # Cập nhật pin: lưu trữ một phần clean energy (hệ số sạc 0.1).
        self.battery_level = min(self.battery_capacity, self.battery_level + clean_energy_generation * 0.1)
        return clean_energy_generation * 0.9, solar_generation, wind_generation
        
    def get_available_energy(self):
        return self.battery_level
    
    def use_battery(self, ratio = 0.2) -> float:
        amount = self.battery_capacity * ratio
        amount_less = max(0, self.battery_level - amount)
        battery_cache = self.battery_level
        self.battery_level = amount_less
        return battery_cache - amount_less