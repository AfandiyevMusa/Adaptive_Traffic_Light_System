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
    capacity = (lane_length / avg_vehicle_length)
    return capacity

# Function to calculate the V/C ratio for all lanes
def get_vc_ratios(lane_ids):
    ratios = []
    for laneID in lane_ids:
        num_vehicles = traci.lane.getLastStepVehicleNumber(laneID)
        capacity = calculate_capacity(laneID)
        ratio = num_vehicles / capacity if capacity > 0 else 0
        ratios.append(ratio)
    return ratios

# Q-value update function
def update_q_table(state, action, reward, next_state, possible_actions, alpha=0.9, gamma=0.1):
    current_q_value = Q_table.get((state, action), 0)  # Get current Q-value
    max_future_q_value = max([Q_table.get((next_state, a), 0) for a in possible_actions])  # Max Q-value for next state
    
    # Q-learning update rule
    new_q_value = current_q_value + alpha * (reward + gamma * max_future_q_value - current_q_value)
    Q_table[(state, action)] = new_q_value

# Reward calculation based on V/C ratio change (for the local and neighbor impact)
def calculate_marl_reward(prev_ratios, new_ratios):
    reward = 0
    for prev_ratio, new_ratio in zip(prev_ratios, new_ratios):
        difference = prev_ratio - new_ratio
        if (0 <= difference <= 0.02):
            reward += 1
        elif (0.02 < difference <= 0.04):
            reward += 2
        elif (0.04 < difference <= 0.06):
            reward += 3
        elif (0.06 < difference <= 0.08):
            reward += 4
        elif (0.08 < difference <= 0.1):
            reward += 5
        elif (-0.02 <= difference <= -0.1):
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
        return random.choice(possible_actions)
    else:
        q_values = [Q_table.get((state, action), 0) for action in possible_actions]
        max_q_value = max(q_values)
        return possible_actions[q_values.index(max_q_value)]

# Function to adjust traffic light times based on the action
def adjust_traffic_lights(tls_id, action):
    logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
    phases = logic.getPhases()
    
    if action[0] == "increase_green":
        phases[0].duration = min(phases[0].duration + 1, 75)
    elif action[0] == "decrease_green":
        phases[0].duration = max(phases[0].duration - 1, 10)
    elif action[0] == "increase_red":
        phases[2].duration = min(phases[2].duration + 1, 75)
    elif action[0] == "decrease_red":
        phases[2].duration = max(phases[2].duration - 1, 10)
    
    traci.trafficlight.setCompleteRedYellowGreenDefinition(tls_id, logic)

# Simulation configuration
def run_simulation(simulation_time):
    traci.start(['sumo', '-c', 'ATLMS.sumocfg'])
    steps = 0
    total_time = simulation_time
    data = []

    while steps < total_time:
        traci.simulationStep()
        
        for tls_id in traci.trafficlight.getIDList():
            logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)[0]
            neighbors = get_neighbors(tls_id)  # Retrieve neighboring TLS IDs
            
            local_lane_ids = traci.trafficlight.getControlledLanes(tls_id)
            neighbor_lane_ids = [traci.trafficlight.getControlledLanes(n) for n in neighbors]
            
            previous_local_vc_ratios = get_vc_ratios(local_lane_ids)
            previous_neighbor_vc_ratios = [get_vc_ratios(l) for l in neighbor_lane_ids]
            
            max_previous_local = max(previous_local_vc_ratios)
            max_previous_neighbors = [max(ratios) for ratios in previous_neighbor_vc_ratios]

            state = (tls_id, round(max_previous_local, 2), tuple(round(r, 2) for r in max_previous_neighbors))
            
            possible_actions = [
                ("increase_green", tls_id),
                ("decrease_green", tls_id),
                ("increase_red", tls_id),
                ("decrease_red", tls_id),
                ("no_change", tls_id)
            ]
            
            action = choose_action(state, possible_actions)
            
            adjust_traffic_lights(tls_id, action)
            traci.simulationStep()

            new_local_vc_ratios = get_vc_ratios(local_lane_ids)
            new_neighbor_vc_ratios = [get_vc_ratios(l) for l in neighbor_lane_ids]
            
            max_new_local = max(new_local_vc_ratios)
            max_new_neighbors = [max(ratios) for ratios in new_neighbor_vc_ratios]
            
            reward = calculate_marl_reward([max_previous_local] + max_previous_neighbors, 
                                           [max_new_local] + max_new_neighbors)
            
            next_state = (tls_id, round(max_new_local, 2), tuple(round(r, 2) for r in max_new_neighbors))
            
            update_q_table(state, action, reward, next_state, possible_actions)
            
            data.append({
                "training_state": steps,
                "tls_id": tls_id,
                "action": action,
                "tl_durations": [phase.duration for phase in logic.getPhases()],
                "reward": reward,
                "max_previous_local": max_previous_local,
                "max_new_local": max_new_local,
                "neighbors": neighbors,
                "max_previous_neighbors": max_previous_neighbors,
                "max_new_neighbors": max_new_neighbors
            })

        steps += 1

    traci.close()
    df = pd.DataFrame(data)
    return df

# Helper function to get neighbors of a TLS
def get_neighbors(tls_id):
    if tls_id == "JM0":
        return ["JM1"]
    elif tls_id == "JM1":
        return ["JM0", "JM2"]
    elif tls_id == "JM2":
        return ["JM1", "J3"]    
    elif tls_id == "J3":
        return ["JM2"]

# Main function to run the simulation and generate the output table
if __name__ == "__main__":
    simulation_time = 1500
    result_df = run_simulation(simulation_time)
    result_df.to_csv('23_11_MARL.csv', index=False)
    print("Simulation completed. Results saved to 'MARL_SIMULATION_RESULTS.csv'.")
