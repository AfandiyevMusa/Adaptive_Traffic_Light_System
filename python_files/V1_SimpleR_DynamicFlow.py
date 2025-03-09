import os
import sys
import random
import numpy as np
from collections import deque
import time

# For logging results to Excel
import xlsxwriter

# TensorFlow / Keras
import tensorflow as tf
from tensorflow import keras

# Import the route randomization functionality from randomized.py
import randomized

# SUMO / TraCI
os.environ["SUMO_HOME"] = "/usr/share/sumo"  # or your existing path
sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
import traci
import sumolib

##############################################################################
# 1. Utility functions 
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
# 2. SumoEnvironment (Modified for speed)
##############################################################################

class SumoEnvironment:
    def __init__(self, sumo_cfg_path, tls_id, edges, lanes,
                 reward_function='queue_length', max_steps=12000):
        self.sumo_cfg_path   = sumo_cfg_path
        self.tls_id          = tls_id
        self.edges           = edges      # Needed to get road_length
        self.lanes           = lanes
        self.reward_function = reward_function
        self.max_steps       = max_steps

        self.conn       = None
        self.step_count = 0

        # Tracking dictionaries
        self.waiting_time_dict     = {lane_id: 0.0 for lane_id in lanes}
        self.last_lane_vehicle_ids = {lane_id: set() for lane_id in lanes}
        self.vehicle_entry_time    = {}

        # For phase duration normalization
        self.min_duration = 1
        self.max_duration = 60

        # New: road_length (static)
        self.road_length = None

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
            "--time-to-teleport", "-1",
            "--route-files", randomized.RANDOMIZED_ROU_FILE
        ]
        traci.start(sumoCmd, label=label)
        self.conn = traci.getConnection(label)
        self.step_count = 0

        for lane_id in self.lanes:
            self.waiting_time_dict[lane_id] = 0.0
            self.last_lane_vehicle_ids[lane_id] = set()
        self.vehicle_entry_time.clear()

        # Retrieve road_length using traci; fallback to sumolib if necessary.
        try:
            self.road_length = self.conn.edge.getLength(self.edges[0])
        except AttributeError:
            print("[Environment] traci.edge.getLength() not available. Falling back to sumolib.")
            net = sumolib.net.readNet("C.net.xml")  # Adjust network file path if needed.
            self.road_length = net.getEdge(self.edges[0]).getLength()

        return self.get_state()

    def step(self, action, sim_time, step_counter):
        """
        5 actions that modify phases[0]=green or phases[1]=red by +/-1 sec
        (action 4 = no change), then advances one simulation step.
        """
        logic_list = self.conn.trafficlight.getAllProgramLogics(self.tls_id)
        if not logic_list:
            raise ValueError(f"No program logic found for {self.tls_id}.")
        logic = logic_list[0]
        phases = logic.getPhases()

        # For two-phase system: phases[0] is green, phases[1] is red.
        g_dur = phases[0].duration
        r_dur = phases[1].duration

        if (step_counter % 2 == 0 and (3500 < sim_time < 4500) or (6500 < sim_time < 7000) or (9000 < sim_time < 10000)):
            g_dur = phases[0].duration
            r_dur = phases[1].duration

            if action == 0:   # green++
                g_dur = min(g_dur + 1, self.max_duration)
            elif action == 1: # green--
                g_dur = max(g_dur - 1, self.min_duration)
            elif action == 2: # red++
                r_dur = min(r_dur + 1, self.max_duration)
            elif action == 3: # red--
                r_dur = max(r_dur - 1, self.min_duration)
            # action=4 => no change

            phases[0].duration = g_dur
            phases[1].duration = r_dur

            # print(f"Sim_time: {sim_time} // case 1 // step_count {step_counter} phase1: {phases[0].duration} phase2: {phases[1].duration}")

            self.conn.trafficlight.setProgramLogic(self.tls_id, logic)

        elif (4501 < sim_time < 5000) or (7001 < sim_time < 7500) or (10001 < sim_time < 10500):
            self.conn.trafficlight.setProgramLogic(self.tls_id, logic)
            # print(f"Sim_time: {sim_time}  // case 2 // step_count {self.step_count}")

        else:
            phases[0].duration = 20
            phases[1].duration = 20
            # print(f"Sim_time:  {sim_time} // case 3 // step_count: {self.step_count}")
            self.conn.trafficlight.setProgramLogic(self.tls_id, logic)

        total_cycle_time = phases[0].duration + phases[1].duration
        total_cycle_time = int(round(total_cycle_time))
        for _ in range(total_cycle_time):
            self.conn.simulationStep()
            self.step_count += 1
            self.update_state_metrics()
            if self.step_count >= self.max_steps:
                break

        reward = self.calculate_reward()
        next_state = self.get_state()

        done = (self.step_count >= self.max_steps or 
                self.conn.simulation.getMinExpectedNumber() <= 0)
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
        Returns a normalized state vector (6 inputs):
          [norm_queue_length, phase0_one_hot, phase1_one_hot, norm_phase_duration, norm_total_vehicles, norm_road_length]
        Normalizations:
          - norm_queue_length = queue_length / 80
          - phase index: one-hot encoded ([1,0] if phase==0, else [0,1])
          - norm_phase_duration = current phase duration / 60
          - norm_total_vehicles = total vehicles / 100
          - norm_road_length = road_length / 100
          - time_one_hot = one_hot_vector(24, sim_time//500), where sim_time is the current simulation time.
        """
        queue_length = 0
        total_veh = 0
        for lane_id in self.lanes:
            queue_length += self.conn.lane.getLastStepHaltingNumber(lane_id)
            total_veh += self.conn.lane.getLastStepVehicleNumber(lane_id)
        norm_queue_length = queue_length / 80.0

        current_phase_idx = self.conn.trafficlight.getPhase(self.tls_id)
        if current_phase_idx == 0:
            phase_one_hot = [1, 0]
        else:
            phase_one_hot = [0, 1]

        logic_list = self.conn.trafficlight.getAllProgramLogics(self.tls_id)
        logic = logic_list[0]
        phases = logic.getPhases()
        phase_dur = phases[current_phase_idx].duration
        norm_phase_dur = phase_dur / 60.0

        norm_total_veh = total_veh / 100.0
        norm_road_length = (self.road_length / 100.0) if self.road_length is not None else 0.0

        sim_time = self.conn.simulation.getTime()

        state_vector = [norm_queue_length] + phase_one_hot + [norm_phase_dur, norm_total_veh, norm_road_length]
        return np.array(state_vector, dtype=float)

    def calculate_reward(self):
        if self.reward_function == 'queue_length':
            total_halts = 0
            for lane_id in self.lanes:
                total_halts += self.conn.lane.getLastStepHaltingNumber(lane_id)
            return -float(total_halts)
        return 0.0

    def close(self):
        try:
            traci.close()
        except:
            pass
        self.conn = None


##############################################################################
# 3. DQNAgent 
##############################################################################

class DQNAgent:
    def __init__(self, state_size, action_size,
                 learning_rate=0.001,
                 gamma=0.8,
                 epsilon=1.0,
                 epsilon_min=0.01,
                 batch_size=128,
                 memory_size=1000):

        self.state_size     = state_size
        self.action_size    = action_size  # 5
        self.learning_rate  = learning_rate
        self.gamma          = gamma
        self.epsilon        = epsilon
        self.epsilon_min    = epsilon_min
        self.batch_size     = batch_size

        self.memory = deque(maxlen=memory_size)

        self.model        = self._build_model()
        self.target_model = self._build_model()
        self._update_target_model()

    def _build_model(self):
        with tf.device("/device:GPU:0"):
            model = keras.Sequential()
            model.add(keras.layers.Dense(64, input_dim=self.state_size, activation='relu'))
            model.add(keras.layers.Dense(64, activation='relu'))
            model.add(keras.layers.Dense(self.action_size, activation='linear'))
            model.compile(
                loss=tf.keras.losses.Huber(),  # Changed loss from 'mse' to Huber loss
                optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate)
            )
        return model

    def _update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # 0..4
        with tf.device("/device:GPU:0"):
            q_values = self.model.predict(state[np.newaxis, :], verbose=0)
        return np.argmax(q_values[0])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)

        states  = np.zeros((self.batch_size, self.state_size))
        targets = np.zeros((self.batch_size, self.action_size))

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            states[i] = state
            with tf.device("/device:GPU:0"):
                target = self.model.predict(state[np.newaxis, :], verbose=0)[0]
                if done:
                    target[action] = reward
                else:
                    t = self.target_model.predict(next_state[np.newaxis, :], verbose=0)[0]
                    target[action] = reward + self.gamma * np.amax(t)
            targets[i] = target

        with tf.device("/device:GPU:0"):
            self.model.fit(states, targets, epochs=1, verbose=0)


##############################################################################
# 4. Training function (V1 style: agent acts every step)
##############################################################################

def train_dqn(sumo_cfg_path, tls_id, edges, lanes, n_episodes, max_steps):
    """
    Runs training over n_episodes. The state is updated at every simulation step.
    The state vector has 6 inputs:
      [norm_queue_length, phase0_one_hot, phase1_one_hot, norm_phase_duration, norm_total_vehicles, norm_road_length]
    """

    # Files for saving weights and episode number
    WEIGHTS_FILE = "dqn_weights.weights.h5"
    EPISODE_FILE = "episode_num.txt"
    EPSILON_FILE = "epsilon.txt"

    # Determine starting episode
    if os.path.exists(EPISODE_FILE):
        with open(EPISODE_FILE, "r") as f:
            try:
                last_episode = int(f.read().strip())
                start_episode = last_episode + 1
                print(f"Resuming from episode {start_episode}")
            except:
                start_episode = 1
    else:
        start_episode = 1

    if os.path.exists(EPSILON_FILE):
        with open(EPSILON_FILE, "r") as f:
            try:
                epsilon_value = float(f.read().strip())
                print(f"Resuming with epsilon = {epsilon_value}")
            except:
                epsilon_value = 1.0
    else:
        epsilon_value = 1.0   

    init_env = SumoEnvironment(
        sumo_cfg_path = sumo_cfg_path,
        tls_id = tls_id,
        edges = edges,
        lanes = lanes,
        reward_function = 'queue_length',
        max_steps = max_steps
    )
    init_state = init_env.reset(episode_idx=0)
    state_size = len(init_state)
    action_size = 5  # [incGreen, decGreen, incRed, decRed, NoChange]

    agent = DQNAgent(
        state_size   = state_size,
        action_size  = action_size,
        gamma        = 0.8,
        epsilon      = epsilon_value,
        batch_size   = 128,
        memory_size  = 1000
    )

    # If a previous weights file exists, load the weights into the agent and update target model
    if os.path.exists(WEIGHTS_FILE):
        agent.model.load_weights(WEIGHTS_FILE)
        agent._update_target_model()
        print("Loaded existing weights.")

    init_env.close()  # Close initial environment instance

     # Prepare Excel logging
    workbook  = xlsxwriter.Workbook('SimpleR_DynamicFlow_Each_Step_07_03.xlsx')
    worksheet = workbook.add_worksheet('Results')

    # We rename "Action" column to "PhaseDurations" as requested
    headers = ["Episode", "Sim_Time", "Step", "PhaseDurations", "QueueLength", "ActiveCars", "Reward"]
    for c, head in enumerate(headers):
        worksheet.write(0, c, head)

    row_excel = 1
    episode_rewards = []

    for e in range(n_episodes):
        # Use the imported function from randomized.py to randomize flows each episode.
        randomized.randomize_flows(randomized.ORIGINAL_ROU_FILE,
                                   randomized.RANDOMIZED_ROU_FILE,
                                   randomized.PREFIX_RANGES)
        print(f"🔄 Flows randomized for episode {e+1}")

        print(f"Starting Episode {e+1}/{n_episodes} current epsilon: {agent.epsilon}")

        # IMPORTANT CHANGE: Create a NEW environment instance per episode so that
        # previous episode data is preserved in agent's memory.
        env = SumoEnvironment(
            sumo_cfg_path, 
            tls_id, 
            edges, 
            lanes, 
            reward_function='queue_length', 
            max_steps=max_steps)
        
        state = env.reset(episode_idx=e)
        total_reward = 0

        for step_counter in range(max_steps):

            # Count active cars
            queue_length = sum(env.conn.lane.getLastStepHaltingNumber(l) for l in lanes)
            active_cars = sum(env.conn.lane.getLastStepVehicleNumber(l) for l in lanes)

            # Retrieve simulation time
            sim_time = env.conn.simulation.getTime()

            # Immediately after step, we retrieve the entire [g,y,r] from the environment
            logic_list = env.conn.trafficlight.getAllProgramLogics(env.tls_id)
            logic      = logic_list[0]
            phases     = logic.getPhases()
            # We'll log them as a string, e.g. "[20, 20]"
            phase_dur = [phases[0].duration, phases[1].duration]

            # Agent decides action in the increasing traffic time
            if  (3500 < sim_time < 4500) or (6500 < sim_time < 7000) or (9000 < sim_time < 10000):
                action = agent.act(state)
                # print(f"TRAIN: Sim_time: {sim_time} // case 1 // step_count: {step_counter}")
            else:
                action = 4  # no change
                # print(f"TRAIN: Sim_time: {sim_time} // case 2 // step_count: {step_counter}")

            next_state, reward, done, _ = env.step(action, sim_time, step_counter)

            # Log episode and step information
            if step_counter % 20 == 0:  # Print every 10 steps
              print(f"Episode {e + 1}, SimulationTime: {sim_time}, Queue_length: {queue_length}, Active_Cars: {active_cars}, Step {step_counter}, Reward: {reward:.2f}, PhaseDuration: {phase_dur}")

            # Log to Excel
            worksheet.write(row_excel, 0, e + start_episode)  # Episode
            worksheet.write(row_excel, 1, sim_time)               # Simulation Time
            worksheet.write(row_excel, 2, step_counter)           # Step
            worksheet.write(row_excel, 3, str(phase_dur))         # Phase durations [G, R]
            worksheet.write(row_excel, 4, queue_length)           # Queue length
            worksheet.write(row_excel, 5, active_cars)            # Active cars
            worksheet.write(row_excel, 6, reward)                 # Reward                             
            row_excel += 1

            # DQN memory & training
            if  (3500 < sim_time < 4500) or (6500 < sim_time < 7000) or (9000 < sim_time < 10000):
                agent.remember(state, action, reward, next_state, done)
                agent.replay()
                if step_counter % 50 == 0 and step_counter > 0:
                    agent._update_target_model()

            state = next_state
            total_reward += reward

            if done:
                print(f"[Episode {e+1}/{n_episodes}] ended early at RL step {step_counter}")
                break

        episode_rewards.append(total_reward)
        print(f"[Episode {e+1}/{n_episodes}] Total Reward: {total_reward:.2f}")

        # Per-episode epsilon decay update:
        agent.epsilon = max(agent.epsilon * 0.95, agent.epsilon_min)
        print(f"Updated epsilon for next episode: {agent.epsilon:.4f}")

        agent._update_target_model()
        env.close()

        # Save current weights and episode number after each episode.
        agent.model.save_weights(WEIGHTS_FILE)
        with open(EPISODE_FILE, "w") as f:
            f.write(str(e + start_episode))  # Save next starting episode
        with open(EPSILON_FILE, "w") as f:
            f.write(str(agent.epsilon))
        print(f"Episode {e + start_episode} and epsilon {agent.epsilon} saved.") 

    # close
    workbook.close()

    avg_reward = np.mean(episode_rewards)
    print(f"\nAverage reward over {n_episodes} episodes: {avg_reward:.2f}")


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
    train_dqn(
        sumo_cfg_path       = SUMO_CFG_PATH,
        tls_id              = my_tls_id,
        edges               = edges,
        lanes               = lanes,
        n_episodes          = 50,
        max_steps           = 12000
    )

