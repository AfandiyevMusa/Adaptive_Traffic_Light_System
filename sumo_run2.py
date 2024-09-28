import traci
import pandas as pd

# Initialize data collection
yellow_state_data = []
yellow_state_count = 0

# Define the relevant lanes at the junction J1
lanes_at_J1 = {
    "-E1_0": "Junction1 -> Edge (-1) -> Lane 1",
    "-E1_1": "Junction1 -> Edge (-1) -> Lane 2",
    "E0_0": "Junction1 -> Edge (0) -> Lane 1",
    "E0_1": "Junction1 -> Edge (0) -> Lane 2"
}

# Start the SUMO simulation using TraCI
def run_simulation(step_limit=180):
    global yellow_state_count

    traci.start(['sumo', '-c', 'ATLMS.sumocfg'])
    
    for step in range(step_limit):
        traci.simulationStep()

        # Check traffic light state at junction J1
        traffic_light_state = traci.trafficlight.getRedYellowGreenState('J1')
        
        if traffic_light_state == "yyyy":  # Yellow light state
            yellow_state_count += 1
            collect_car_data(step)

    traci.close()

# Function to collect data about cars at the relevant lanes during yellow light
def collect_car_data(step):
    global yellow_state_data
    
    for lane_id, lane_description in lanes_at_J1.items():
        num_cars = traci.lane.getLastStepVehicleNumber(lane_id)
        yellow_state_data.append({
            "Yellow state number": yellow_state_count,
            "Time (step)": step,
            "Coordinate": lane_description,
            "Number of cars": num_cars
        })

# Save the collected data to an Excel file
def save_data_to_excel(file_name='number_of_cars_2.xlsx'):
    df = pd.DataFrame(yellow_state_data)
    df.to_excel(file_name, index=False)
    print(f"Data saved to {file_name}")

# Run the simulation and save data to Excel
run_simulation()
save_data_to_excel()
