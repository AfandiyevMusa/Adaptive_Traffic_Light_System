import random
import xml.etree.ElementTree as ET
import traci
import pandas as pd
import matplotlib.pyplot as plt
import datetime

# CONFIGURATION
# The original route file with flows
ORIGINAL_ROU_FILE = "C.rou.xml"

# The new route file to be created with randomized flows
RANDOMIZED_ROU_FILE = "rou_randomized.xml"

# (c) SUMO configuration:
#    If you have a sumocfg file that references rou.xml, you can either
#    update it to reference "rou_randomized.xml" or override it at runtime.
#
#    Method 1 (recommended):
#       - In your sumocfg, under <input><route-files value="rou_randomized.xml"/></input>.
#         Then we can just do:
#             SUMO_CMD = ["sumo", "-c", "C.sumocfg"]
#
#    Method 2:
#       - Or supply route files on the command line:
#             SUMO_CMD = ["sumo", "-c", "C.sumocfg", "--route-files", RANDOMIZED_ROU_FILE]

SUMO_CMD = ["sumo", "-c", "C.sumocfg", "--route-files", RANDOMIZED_ROU_FILE]

TOTAL_SIMULATION_STEPS = 24000  # Simulation steps represent 24-hour period

PREFIX_RANGES = {
    "night1_flow_":         (10,  15),   # 00:00 – 07:00
    "morning_flow_":        (200, 300),  # 07:00 – 09:00
    "mid_morning_flow_":    (140, 180),  # 09:00 – 13:00
    "lunch_flow_":          (120, 160),  # 13:00 – 14:00
    "afternoon_flow_":      (140, 180),  # 14:00 – 18:00
    "evening_flow_":        (200, 300),  # 18:00 – 20:00
    "night_flow_":          (140, 180)   # 20:00 – 00:00
}

# (Optional) Fix the random seed if you want the same numbers each run:
# random.seed(42)

# RANDOMIZE THE FLOWS
def randomize_flows(input_file, output_file, prefix_ranges):
    tree = ET.parse(input_file)
    root = tree.getroot()

    for flow in root.findall('flow'):
        flow_id = flow.get('id', '')
        if 'number' in flow.attrib:
            assigned_range = None
            for prefix, (low, high) in prefix_ranges.items():
                if flow_id.startswith(prefix):
                    assigned_range = (low, high)
                    break
            if assigned_range is not None:
                low, high = assigned_range
                new_number = random.randint(low, high)
                flow.set('number', str(new_number))

    tree.write(output_file, encoding='UTF-8', xml_declaration=True)
    print(f"✅ Randomized flows saved to {output_file}")

# Convert simulation step to time format (HH:MM), ensuring rounded times
def step_to_time(step, total_steps=TOTAL_SIMULATION_STEPS):
    start_time = datetime.datetime.strptime("00:00", "%H:%M")
    end_time = datetime.datetime.strptime("23:59", "%H:%M")
    
    time_diff = end_time - start_time
    time_at_step = start_time + (time_diff * (step / total_steps))

    # Round minutes to nearest full hour or 5-minute interval
    rounded_time = time_at_step.replace(second=0, microsecond=0)
    if rounded_time.minute >= 30:
        rounded_time += datetime.timedelta(minutes=(60 - rounded_time.minute))
    else:
        rounded_time -= datetime.timedelta(minutes=rounded_time.minute)

    return rounded_time.strftime("%H:%M")

# RUN SUMO AND COLLECT DATA
def run_sumo_and_record(sumo_cmd):
    traci.start(sumo_cmd)
    step = 0

    simulation_steps = []
    active_vehicles = []

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        num_active_vehicles = traci.vehicle.getIDCount()

        simulation_steps.append(step)
        active_vehicles.append(num_active_vehicles)

        step += 1

    traci.close()

    df = pd.DataFrame({"Simulation Step": simulation_steps, "Active Vehicles": active_vehicles})
    df.to_excel("traffic_flow_record.xlsx", index=False)
    print("✅ Traffic data saved to traffic_flow_record.xlsx")

    return simulation_steps, active_vehicles

# MAIN ENTRY POINT
if __name__ == "__main__":
    randomize_flows(ORIGINAL_ROU_FILE, RANDOMIZED_ROU_FILE, PREFIX_RANGES)
    
    # Run SUMO and get simulation data
    simulation_steps, active_vehicles = run_sumo_and_record(SUMO_CMD)

    # Define major tick positions (e.g., every 5000 steps)
    major_ticks = list(range(0, TOTAL_SIMULATION_STEPS + 1, 5000))

    # Convert major tick steps to time labels
    time_labels = [step_to_time(step) for step in major_ticks]

    # Create a plot with both simulation step numbers and corresponding times
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Scatter plot for active vehicles
    ax1.scatter(simulation_steps, active_vehicles, color="blue", label="Active Cars")
    ax1.set_xlabel("Simulation Step (Time of Day)")
    ax1.set_ylabel("Number of Active Cars")
    ax1.set_title("Traffic Flow Over Time")
    ax1.legend()
    ax1.grid(True)

    # Set custom x-axis ticks
    ax1.set_xticks(major_ticks)
    ax1.set_xticklabels([f"{step}\n{time}" for step, time in zip(major_ticks, time_labels)], rotation=0, fontsize=10)

    plt.show()
