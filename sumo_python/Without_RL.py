import sys
sys.path.append('/Users/rufatismayilov/sumo/tools')
import traci
import numpy as np
import pandas as pd
print(pd.__version__)
import random

epsilon = 0.9  # Exploration rate
Q_table = {}

# Function to calculate lane capacity
def calculate_capacity(laneID):
    lane_length = traci.lane.getLength(laneID)
    avg_vehicle_length = 6  # Average vehicle length in meters
    capacity = (lane_length / avg_vehicle_length) # 350 / 7 = 50 
    return capacity

# Function to calculate the V/C ratio for all lanes
def get_vc_ratios(lane_ids):
    ratios = []
    for laneID in lane_ids:
        num_vehicles = traci.lane.getLastStepVehicleNumber(laneID)
        capacity = calculate_capacity(laneID)
        ratio = num_vehicles / capacity if capacity > 0 else 0 # 30 / 50
        ratios.append(ratio)
    return ratios

# Q-value update function
def update_q_table(state, action, reward, next_state, possible_actions, alpha=0.9, gamma=0.1):
    current_q_value = Q_table.get((state, action), 0)  # Get current Q-value
    max_future_q_value = max([Q_table.get((next_state, a), 0) for a in possible_actions])  # Get max Q-value for next state
    
    # Q-learning update rule
    new_q_value = current_q_value + alpha * (reward + gamma * max_future_q_value - current_q_value)
    Q_table[(state, action)] = new_q_value

# Reward calculation based on V/C ratio change
def calculate_reward(prev_ratio, new_ratio):
    reward = 0
    difference = prev_ratio - new_ratio
    if (0 <= difference <= 0.02):  # Improvement (ratio decreases)
        reward += 1
    elif (0.02 < difference <= 0.04):
        reward += 2
    elif (0.04 < difference <= 0.06):
        reward += 3
    elif (0.06 < difference <= 0.08):
        reward += 4 
    elif (0.08 < difference <= 0.1):
        reward += 5
    elif (-0.02 <= difference <= -0.1):  # Punishment (ratio increases)
        reward -= 1
    elif (-0.04 <= difference < -0.02):
        reward -= 2
    elif (-0.06 <= difference < -0.04):
        reward -= 3
    elif (-0.08 <= difference < -0.06):
        reward -= 4 
    elif (-0.1 <= difference < -0.08):
        reward -= 5                
    return reward

# Function to choose an action based on an epsilon-greedy strategy
def choose_action(state, possible_actions):
    if random.uniform(0, 1) < epsilon:
        # Exploration: choose a random action
        return random.choice(possible_actions)
    else:
        # Exploitation: choose the action with the highest Q-value
        q_values = [Q_table.get((state, action), 0) for action in possible_actions]
        max_q_value = max(q_values)
        return possible_actions[q_values.index(max_q_value)]


# Function to adjust traffic light times based on the action
def adjust_traffic_lights(tls_id, action):
    logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]  # Updated to avoid deprecated function
    phases = logic.getPhases()
    
    # Modify durations based on action with boundary conditions
    if action[0] == "increase_green":
        phases[0].duration = min(phases[0].duration + 1, 75)  # Increase green phase, max 75 seconds
    elif action[0] == "decrease_green":
        phases[0].duration = max(phases[0].duration - 1, 10)  # Decrease green phase, min 10 seconds
    elif action[0] == "increase_red":
        phases[2].duration = min(phases[2].duration + 1, 75)  # Increase red phase, max 75 seconds
    elif action[0] == "decrease_red":
        phases[2].duration = max(phases[2].duration - 1, 10)  # Decrease red phase, min 10 seconds
    # "No change" action leaves the durations unchanged
    
    # Apply the updated logic
    traci.trafficlight.setCompleteRedYellowGreenDefinition(tls_id, logic)

# Simulation configuration
def run_simulation(simulation_time):
    traci.start(['sumo', '-c', 'ATLMS.sumocfg'])
    steps = 0
    total_time = simulation_time
    
    data = []  # To store each training step information

    while steps < total_time:
        traci.simulationStep()  # Advance the simulation by one step
        
        # Extract data at every yellow light phase change
        for tls_id in traci.trafficlight.getIDList():    

            # if tls_id != "JM1":
            #     continue

            logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)[0]
            for phase in logic.getPhases():
                if 'y' in phase.state:  # Yellow light phase
                    lane_ids = traci.trafficlight.getControlledLanes(tls_id)
                    # Get the number of vehicles and lane capacities
                    previous_vc_ratios = get_vc_ratios(lane_ids)
                    
                    # **Update 1**: Use the maximum V/C ratio instead of average
                    max_previous_ratio = max(previous_vc_ratios)  # Use max V/C ratio
                    
                    # Define the current state (for simplicity, we use the max ratio and tls_id)
                    state = (tls_id, round(max_previous_ratio, 2))
                    
                    # Possible actions: include a "No change" option
                    possible_actions = [
                        ("increase_green", tls_id),
                        ("decrease_green", tls_id),
                        ("increase_red", tls_id),
                        ("decrease_red", tls_id),
                        ("no_change", tls_id)
                    ]
                    
                    # Choose an action using the epsilon-greedy strategy
                    action = choose_action(state, possible_actions)
                    
                    # Adjust traffic light durations based on the action
                    adjust_traffic_lights(tls_id, action)
                    
                    # Run for a step to see the effect of the action
                    traci.simulationStep()

                    # Get new V/C ratios
                    new_vc_ratios = get_vc_ratios(lane_ids)
                    
                    # **Update 2**: Use the maximum V/C ratio for the new state
                    max_new_ratio = max(new_vc_ratios)  # Use max V/C ratio
                    reward = calculate_reward(max_previous_ratio, max_new_ratio)
                    
                    # Define the new state after taking the action
                    next_state = (tls_id, round(max_new_ratio, 2))
                    
                    # Update the Q-table
                    update_q_table(state, action, reward, next_state, possible_actions)
                    
                    # Store the traffic light timings and state info
                    data.append({
                        "training_state": steps,
                        "yellow_state": phase.state,
                        "lane_ids": lane_ids,
                        "#of_cars": [traci.lane.getLastStepVehicleNumber(l) for l in lane_ids],
                        "ratio": previous_vc_ratios,
                        "max_ratio": max_previous_ratio,  # Store max ratio instead of avg
                        "new_max_ratio": max_new_ratio,   # Store new max ratio
                        "tl_durations": [phase.duration for phase in logic.getPhases()],
                        "tl_id": tls_id,
                        "action": action,
                        "reward": reward
                    })

        steps += 1  # Move to the next simulation step

    traci.close()  # Close the SUMO simulation

    # Convert collected data to a DataFrame for easy manipulation and export 
    df = pd.DataFrame(data)
    return df

    # Main function to run the simulation and generate the output table
if __name__ == "__main__":
    simulation_time = 1500  # Total simulation time in seconds
    result_df = run_simulation(simulation_time)

    # Save the result to a CSV or display it
    result_df.to_csv('01_12_WithRL.csv', index=False)
    print("Simulation completed. Results saved to 'output.csv'.")
