import numpy as np
import os

'''
EnergySystem class:
* solar_capacity: maximum solar energy generation capacity.
* wind_capacity: maximum wind energy generation capacity.
* battery_capacity: clean energy storage capacity.
* battery_level: current battery level.
'''
class EnergySystem:
    def __init__(self, solar_capacity, wind_capacity, battery_capacity, time_step, train = False):
        self.solar_capacity = solar_capacity   
        self.wind_capacity = wind_capacity      
        self.battery_capacity = battery_capacity  
        self.battery_level = battery_capacity 
        
        # if file exists, do not generate new data
        if train == False:
            if not os.path.exists('energy_data.csv'):
                self.generate_energy(time_step)
            else:
                self.energy_data = []
                with open('energy_data.csv', 'r') as f:
                    f.readline()
                    for line in f:
                        data = list(map(float, line.strip().split(',')))
                        self.energy_data.append(data)
        else:
            self.generate_energy(time_step, train=True)

    def generate_energy(self, time_step, train = False):
        # Generate energy data for 24 hours.
        self.energy_data = []
        for h in range(time_step):
            hour = h % 24
            solar_generation = self.solar_capacity * np.clip(np.sin(np.pi * (hour - 6) / 12), 0, None)
            wind_generation = self.wind_capacity * (0.5 + 0.3 * np.sin(np.pi * hour / 24) + np.random.uniform(-1, 1))
            wind_generation = max(wind_generation, 0)
            clean_energy_generation = solar_generation + wind_generation
            self.energy_data.append([clean_energy_generation, solar_generation, wind_generation])
        # save to file
        if train == False:
            with open('energy_data.csv', 'w') as f:
                f.write('clean_energy,solar_energy,wind_energy\n')
                for data in self.energy_data:
                    f.write(','.join(map(str, data)) + '\n')

    def get_energy_data(self, time):
        return self.energy_data[time]
        
    def update(self, time_step):
        solar_generation, wind_generation, clean_energy_generation = self.get_energy_data(time_step)
        
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