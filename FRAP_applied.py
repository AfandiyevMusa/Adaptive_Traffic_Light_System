import traci
import sumolib
import numpy as np
import pandas as pd
import random

# Q-learning parameters
alpha = 0.9  # Learning rate
gamma = 0.1  # Discount factor
epsilon = 0.9  # Exploration rate

# Initialize Q-table as a dictionary: {(state, action): Q-value}
Q_table = {}

# Function to calculate the capacity of a lane
def calculate_capacity(laneID):
    lane_length = traci.lane.getLength(laneID)
    avg_vehicle_length = 5  # Average vehicle length in meters
    num_lanes = traci.lane.getWidth(laneID)  # Assuming lane width translates to lane count
    capacity = (lane_length / avg_vehicle_length) * num_lanes
    return capacity

# Reward function based on V/C ratio improvement
def calculate_reward(new_vc_ratios):
    avg_vc_ratio = np.mean(new_vc_ratios)  # Take the average of the V/C ratios
    return -avg_vc_ratio  # Negative reward to encourage minimizing V/C ratio

# Function to calculate the V/C ratio for all lanes
def get_vc_ratios(lane_ids):
    ratios = []
    for laneID in lane_ids:
        num_vehicles = traci.lane.getLastStepVehicleNumber(laneID)
        capacity = calculate_capacity(laneID)
        ratio = num_vehicles / capacity if capacity > 0 else 0
        ratios.append(ratio)
    return ratios

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

# Q-learning update function
def update_q_table(state, action, reward, next_state, next_actions):
    current_q_value = Q_table.get((state, action), 0)
    future_q_values = [Q_table.get((next_state, next_action), 0) for next_action in next_actions]
    max_future_q_value = max(future_q_values) if future_q_values else 0

    # Q-learning formula
    new_q_value = current_q_value + alpha * (reward + gamma * max_future_q_value - current_q_value)
    Q_table[(state, action)] = new_q_value

# Dynamic adjustment based on Q-value difference
def calculate_dynamic_change(state, action):
    q_values = Q_table.get(state, np.zeros(4))
    max_q_value = np.max(q_values)
    current_q_value = q_values[action]

    # Proportional to Q-value difference
    q_diff = max_q_value - current_q_value if max_q_value != 0 else 0
    dynamic_change = 1 + 5 * (q_diff / max_q_value) if max_q_value != 0 else 1

    return dynamic_change

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
            logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)[0]
            for phase in logic.getPhases():
                if 'y' in phase.state:  # Yellow light phase
                    lane_ids = traci.trafficlight.getControlledLanes(tls_id)
                    
                    # Get the number of vehicles and lane capacities
                    vc_ratios = get_vc_ratios(lane_ids)
                    avg_ratio = np.mean(vc_ratios)
                    
                    # Define the current state (for simplicity, we use the average ratio and tls_id)
                    state = (tls_id, round(avg_ratio, 2))
                    
                    # Possible actions: we can increase/decrease the duration of green, yellow, or red phases
                    possible_actions = [
                        ("increase_green", tls_id),
                        ("decrease_green", tls_id),
                        ("increase_red", tls_id),
                        ("decrease_red", tls_id)
                    ]
                    
                    # Choose an action using the epsilon-greedy strategy
                    action = choose_action(state, possible_actions)
                    
                    # Adjust traffic light durations based on the action and Q-values
                    dynamic_change = calculate_dynamic_change(state, possible_actions.index(action))
                    adjust_traffic_lights(tls_id, action, dynamic_change)
                    
                    # Run for a step to see the effect of the action
                    traci.simulationStep()

                    # Calculate the reward based on the new V/C ratios
                    new_vc_ratios = get_vc_ratios(lane_ids)
                    reward = calculate_reward(new_vc_ratios)
                    
                    # Define the new state after taking the action
                    next_state = (tls_id, round(np.mean(new_vc_ratios), 2))
                    
                    # Update the Q-table
                    update_q_table(state, action, reward, next_state, possible_actions)
                    
                    # Store the traffic light timings and state info
                    data.append({
                        "training_state": steps,
                        "yellow_state": phase.state,
                        "lane_ids": lane_ids,
                        "#of_cars": [traci.lane.getLastStepVehicleNumber(l) for l in lane_ids],
                        "ratio": vc_ratios,
                        "avg_ratio": avg_ratio,
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

# Function to adjust traffic light times based on the action and dynamic change
def adjust_traffic_lights(tls_id, action, dynamic_change):
    logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)[0]
    phases = logic.getPhases()
    
    # Iterate over the phases to find the ones without yellow ('y')
    for i, phase in enumerate(phases):
        if 'y' in phase.state:
            continue  # Skip the phase if it contains yellow

        # Modify the duration of the specific phase based on the action and dynamic_change
        if action[0] == "increase_green" and 'G' in phase.state:
            phases[i].duration += dynamic_change  # Increase green phase dynamically
        elif action[0] == "decrease_green" and 'G' in phase.state:
            phases[i].duration = max(0, phases[i].duration - dynamic_change)  # Decrease green phase
        elif action[0] == "increase_red" and 'r' in phase.state:
            phases[i].duration += dynamic_change  # Increase red phase dynamically
        elif action[0] == "decrease_red" and 'r' in phase.state:
            phases[i].duration = max(0, phases[i].duration - dynamic_change)  # Decrease red phase    

    # Apply the updated logic
    traci.trafficlight.setCompleteRedYellowGreenDefinition(tls_id, logic)

# Main function to run the simulation and generate the output table
if __name__ == "__main__":
    simulation_time = 180  # Total simulation time in seconds
    result_df = run_simulation(simulation_time)

    # Save the result to a CSV or display it
    result_df.to_csv('FRAP_applied1.csv', index=False)
    print("Simulation completed. Results saved to 'output.csv'.")
