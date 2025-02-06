import os
import sys
import random
import numpy as np
from collections import deque

# For logging results to Excel
import xlsxwriter

# TensorFlow / Keras
import tensorflow as tf
from tensorflow import keras

# SUMO / TraCI
os.environ["SUMO_HOME"] = "/usr/share/sumo"  # or your existing path
sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
import traci
import sumolib


##############################################################################
# 1. Utility functions (unchanged)
##############################################################################

def get_sumo_info(sumo_cfg_path, label="infoConnection"):
    sumoCmd = [
        "sumo",
        "-c", sumo_cfg_path,
        "--no-step-log", "true",
        "--time-to-teleport", "-1"
    ]
    traci.start(sumoCmd, label=label)
    conn = traci.getConnection(label)

    tls_ids  = conn.trafficlight.getIDList()
    edge_ids = conn.edge.getIDList()
    lane_ids = conn.lane.getIDList()

    conn.close()
    return tls_ids, edge_ids, lane_ids


def get_tls_phases(sumo_cfg_path, tls_id, label="phaseConnection"):
    sumoCmd = [
        "sumo",
        "-c", sumo_cfg_path,
        "--no-step-log", "true",
        "--time-to-teleport", "-1"
    ]
    traci.start(sumoCmd, label=label)
    conn = traci.getConnection(label)

    logic_list = conn.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)
    phase_dict = {}
    if logic_list:
        logic = logic_list[0]
        for i, phase_obj in enumerate(logic.phases):
            phase_dict[i] = int(phase_obj.duration)
    else:
        print(f"No logic found for TLS {tls_id}.")

    conn.close()
    return phase_dict


##############################################################################
# 2. SUMO Environment (5 actions: inc/dec green, inc/dec red, no change)
##############################################################################

class SumoEnvironment:
    def __init__(self, sumo_cfg_path, tls_id, edges, lanes,
                 reward_function='intellilight', max_steps=3600):
        self.sumo_cfg_path   = sumo_cfg_path
        self.tls_id          = tls_id
        self.edges           = edges
        self.lanes           = lanes
        self.reward_function = reward_function
        self.max_steps       = max_steps

        self.conn       = None
        self.step_count = 0

        # Track waiting time, vehicles, etc.
        self.waiting_time_dict    = {lane_id: 0.0 for lane_id in lanes}
        self.last_lane_vehicle_ids= {lane_id: set() for lane_id in lanes}

        self.speed_limit_dict     = {}
        self._default_speed_limit = 25.9  # ~50 km/h
        self.vehicle_entry_time   = {}

        # IntelliLight-like reward weights
        self.w1, self.w2, self.w3 = -0.25, -0.25, -0.25
        self.w4, self.w5, self.w6 = -5, 1, 1

        # Duration clamp
        self.min_duration = 1
        self.max_duration = 100

    def reset(self, episode_idx=0):
        try:
            traci.close(False)
        except traci.FatalTraCIError:
            pass

        label = f"trainConnection_{episode_idx}"
        sumoCmd = [
            "sumo",
            "-c", self.sumo_cfg_path,
            "--no-step-log", "true",
            "--time-to-teleport", "-1"
        ]
        traci.start(sumoCmd, label=label)
        self.conn = traci.getConnection(label)

        self.step_count = 0

        for lane_id in self.lanes:
            try:
                speedlim = self.conn.lane.getMaxSpeed(lane_id)
                self.speed_limit_dict[lane_id] = speedlim
            except:
                self.speed_limit_dict[lane_id] = self._default_speed_limit

        for lane_id in self.lanes:
            self.waiting_time_dict[lane_id] = 0.0
            self.last_lane_vehicle_ids[lane_id] = set()
        self.vehicle_entry_time.clear()

        return self.get_state()

    def step(self, action):
        """
        5 actions that modify phases[0]=green or phases[2]=red by +/-1 sec
        or do nothing (action=4). Then run G+Y+R steps.
        """
        logic_list = self.conn.trafficlight.getAllProgramLogics(self.tls_id)
        if not logic_list:
            raise ValueError(f"No program logic found for {self.tls_id}.")
        logic  = logic_list[0]
        phases = logic.getPhases()

        g_dur = phases[0].duration
        r_dur = phases[2].duration

        self.conn.trafficlight.setProgramLogic(self.tls_id, logic)

        total_cycle_time = phases[0].duration + phases[1].duration + phases[2].duration
        total_cycle_time = int(round(total_cycle_time))
        for _ in range(total_cycle_time):
            self.conn.simulationStep()
            self.step_count += 1
            self.update_state_metrics()
            if self.step_count >= self.max_steps:
                break

        reward = self.calculate_reward(action)
        next_state = self.get_state()

        done = (
            self.step_count >= self.max_steps
            or self.conn.simulation.getMinExpectedNumber() <= 0
        )
        return next_state, reward, done, {}

    def update_state_metrics(self):
        sim_time = self.conn.simulation.getTime()
        for lane_id in self.lanes:
            veh_ids = set(self.conn.lane.getLastStepVehicleIDs(lane_id))
            waiting_count = 0
            for vid in veh_ids:
                speed = self.conn.vehicle.getSpeed(vid)
                if speed < 0.1:
                    waiting_count += 1
            self.waiting_time_dict[lane_id] += waiting_count

            old_set = self.last_lane_vehicle_ids[lane_id]
            new_ones = veh_ids - old_set
            for vid in new_ones:
                if vid not in self.vehicle_entry_time:
                    self.vehicle_entry_time[vid] = sim_time
            self.last_lane_vehicle_ids[lane_id] = veh_ids

    def get_state(self):
        """
        [queue_length, phase_index, phase_duration, waiting_time, total_vehicles]
        """
        queue_length = 0
        total_veh    = 0
        for lane_id in self.lanes:
            queue_length += self.conn.lane.getLastStepHaltingNumber(lane_id)
            total_veh    += self.conn.lane.getLastStepVehicleNumber(lane_id)

        current_phase_idx = self.conn.trafficlight.getPhase(self.tls_id)
        logic_list = self.conn.trafficlight.getAllProgramLogics(self.tls_id)
        logic      = logic_list[0]
        phases     = logic.getPhases()
        phase_dur  = phases[current_phase_idx].duration

        waiting_time = sum(self.waiting_time_dict[l] for l in self.lanes)

        state = [queue_length, current_phase_idx, phase_dur, waiting_time, total_veh]
        return np.array(state, dtype=float)

    def calculate_reward(self, action):
        """
        IntelliLight 6-term
        R = w1 * sum(L) + w2 * sum(D) + w3 * sum(W) + w4 * C + w5*N + w6*T
        """
        sim_time = self.conn.simulation.getTime()

        # sum(L)
        total_queue = 0
        for lane_id in self.lanes:
            total_queue += self.conn.lane.getLastStepHaltingNumber(lane_id)

        # sum(D)
        total_delay = 0.0
        for lane_id in self.lanes:
            veh_ids = self.conn.lane.getLastStepVehicleIDs(lane_id)
            if len(veh_ids) == 0:
                continue
            sum_speed = 0.0
            for vid in veh_ids:
                sum_speed += self.conn.vehicle.getSpeed(vid)
            avg_speed  = sum_speed / len(veh_ids)

            speed_lim  = self.speed_limit_dict[lane_id]
            d_i = 1.0 - (avg_speed / speed_lim)
            if d_i < 0:
                d_i = 0
            total_delay += d_i

        # sum(W) ~ step_waiting
        step_waiting = 0
        for lane_id in self.lanes:
            veh_ids = self.conn.lane.getLastStepVehicleIDs(lane_id)
            for vid in veh_ids:
                if self.conn.vehicle.getSpeed(vid) < 0.1:
                    step_waiting += 1

        # C => if action in [0,1,2,3], then 1; if 4 => 0
        C = 1 if action in [0,1,2,3] else 0

        # N => vehicles that left
        passed_count = 0
        passed_travel_time = 0.0
        for vid in list(self.vehicle_entry_time.keys()):
            still_in_lanes = False
            for lane_id in self.lanes:
                if vid in self.last_lane_vehicle_ids[lane_id]:
                    still_in_lanes = True
                    break
            if not still_in_lanes:
                passed_count += 1
                start_t = self.vehicle_entry_time[vid]
                travel_secs = sim_time - start_t
                passed_travel_time += travel_secs
                del self.vehicle_entry_time[vid]

        # T => minutes
        T = passed_travel_time / 60.0

        r = (self.w1 * total_queue) \
          + (self.w2 * total_delay) \
          + (self.w3 * step_waiting) \
          + (self.w4 * C) \
          + (self.w5 * passed_count) \
          + (self.w6 * T)

        return r

    def close(self):
        try:
            traci.close()
        except:
            pass
        self.conn = None


##############################################################################
# 4. Training function
##############################################################################

def train_dqn_fixed_steps(sumo_cfg_path, tls_id, edges, lanes,
                          n_episodes=5, max_steps=3000,
                          rl_steps_per_episode=50):
    """
    We log the *phase durations* in place of the old "Action" column.
    """
    env = SumoEnvironment(
        sumo_cfg_path   = sumo_cfg_path,
        tls_id          = tls_id,
        edges           = edges,
        lanes           = lanes,
        reward_function = 'intellilight',
        max_steps       = max_steps
    )

    init_state = env.reset(episode_idx=0)
    state_size = len(init_state)
    action_size = 5  # incGreen, decGreen, incRed, decRed, noChange

    # Prepare Excel logging
    workbook  = xlsxwriter.Workbook('WithoutRL_06_02.xlsx')
    worksheet = workbook.add_worksheet('Results')

    # We rename "Action" column to "PhaseDurations" as requested
    headers = ["Episode","Step","State","PhaseDurations","ActiveCars","Reward"]
    for c, head in enumerate(headers):
        worksheet.write(0, c, head)

    row_excel = 1
    episode_rewards = []
    update_model_interval = 300

    for e in range(n_episodes):
        print(f"Starting Episode {e + 1}/{n_episodes}")
        state = env.reset(episode_idx=e)
        total_reward = 0

        for step_counter in range(rl_steps_per_episode):

            next_state, reward, done, _ = env.step(0)

            # Immediately after step, we retrieve the entire [g,y,r] from the environment
            logic_list = env.conn.trafficlight.getAllProgramLogics(env.tls_id)
            logic      = logic_list[0]
            phases     = logic.getPhases()
            # We'll log them as a string, e.g. "[60, 3, 15]"
            current_phase_durations = [phases[0].duration,
                                       phases[1].duration,
                                       phases[2].duration]

            # Count active cars
            active_cars = 0
            for lane_id in lanes:
                active_cars += env.conn.lane.getLastStepVehicleNumber(lane_id)

            # Write to Excel
            worksheet.write(row_excel, 0, e)                      # Episode
            worksheet.write(row_excel, 1, step_counter)           # Step
            worksheet.write(row_excel, 2, str(state.tolist()))    # State
            worksheet.write(row_excel, 3, str(current_phase_durations))
            worksheet.write(row_excel, 4, active_cars)
            worksheet.write(row_excel, 5, reward)
            row_excel += 1

            state = next_state

    # close
    workbook.close()
    env.close()

##############################################################################
# 5. MAIN
##############################################################################

if __name__ == "__main__":
    SUMO_CFG_PATH = "C.sumocfg"
    print("Testing SUMO config offline:")
    # !sumo -c $SUMO_CFG_PATH

    tls_ids, edges, lanes = get_sumo_info(SUMO_CFG_PATH, label="infoConnection")
    print("TLS:", tls_ids)
    print("Edges:", edges)
    print("Lanes:", lanes)

    if not tls_ids:
        raise ValueError("No traffic lights found.")
    my_tls_id = tls_ids[0]

    phases = get_tls_phases(SUMO_CFG_PATH, my_tls_id, label="phaseConnection")
    print("Initial phases:", phases)

    # run training
    train_dqn_fixed_steps(
        sumo_cfg_path       = SUMO_CFG_PATH,
        tls_id              = my_tls_id,
        edges               = edges,
        lanes               = lanes,
        n_episodes          = 1,
        max_steps           = 3000,
        rl_steps_per_episode= 100
    )
