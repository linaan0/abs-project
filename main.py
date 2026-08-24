import cityflow

config_path = "/CityFlow/examples/config.json"

eng = cityflow.Engine(config_path, thread_num=1)

for _ in range(100):
    eng.next_step()

print("Simulation finished!")
print("Simulation time:", eng.get_current_time())
print("Vehicles:", eng.get_vehicle_count())
print("Average travel time:", eng.get_average_travel_time())