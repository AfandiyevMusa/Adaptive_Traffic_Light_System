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
os.environ["SUMO_HOME"] = "/usr/share/sumo"  # Adjust if needed
sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
import traci
import sumolib


########################################################################
# 1. Utility functions
########################################################################

def get_sumo_info(sumo_cfg_path, label="infoConnection"):
    """
    1) Starts a short SUMO run with a custom label to query network data,
    2) Returns (tls_ids, edge_ids, lane_ids),
    3) Closes that connection.
    """
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
    """
    Retrieves phase durations from SUMO for a given traffic light ID.
    Returns {phase_index: duration} just for reference.
    """
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


########################################################################
# 2. SUMO Environment
########################################################################

class SumoEnvironment:
    """
    A 3-phase cycle: [phase0=green, phase1=yellow, phase2=red].
    Actions:
      0 -> incGreen
      1 -> decGreen
      2 -> incRed
      3 -> decRed
    """
    def __init__(self, sumo_cfg_path, tls_id, edges, lanes,
                 reward_function='queue_length', max_steps=3600):
        self.sumo_cfg_path   = sumo_cfg_path
        self.tls_id          = tls_id
        self.edges           = edges
        self.lanes           = lanes
        self.reward_function = reward_function
        self.max_steps       = max_steps

        self.step_count = 0
        self.conn       = None

        # min/max durations for green or red
        self.min_duration = 1
        self.max_duration = 120

    def reset(self, episode_idx=0):
        """
        Reset SUMO from time=0 for a new episode.
        """
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
        return self.get_state()

    def step(self, action):
        """
        1) Adjust green/red durations
        2) Run (green+yellow+red) steps
        3) Compute reward, next_state, done
        """
        logic_list = self.conn.trafficlight.getAllProgramLogics(self.tls_id)
        if not logic_list:
            raise ValueError(f"No program logic found for {self.tls_id}.")
        logic  = logic_list[0]
        phases = logic.getPhases()
        if len(phases) < 3:
            raise ValueError("Expected at least 3 phases: [green, yellow, red].")

        # Current durations
        green_dur = phases[0].duration
        red_dur   = phases[2].duration

        # Adjust durations
        if action == 0:  # incGreen
            green_dur = min(green_dur + 1, self.max_duration)
        elif action == 1:  # decGreen
            green_dur = max(green_dur - 1, self.min_duration)
        elif action == 2:  # incRed
            red_dur   = min(red_dur + 1, self.max_duration)
        elif action == 3:  # decRed
            red_dur   = max(red_dur - 1, self.min_duration)

        phases[0].duration = green_dur
        # phases[1] (yellow) is unchanged
        phases[2].duration = red_dur

        self.conn.trafficlight.setProgramLogic(self.tls_id, logic)

        total_cycle_time = phases[0].duration + phases[1].duration + phases[2].duration
        for _ in range(int(total_cycle_time)):
            self.conn.simulationStep()
            self.step_count += 1
            if self.step_count >= self.max_steps:
                break

        reward = self.calculate_reward()
        next_state = self.get_state()

        done = (
            self.step_count >= self.max_steps
            or self.conn.simulation.getMinExpectedNumber() <= 0
        )

        current_phases = [phases[0].duration, phases[1].duration, phases[2].duration]
        return next_state, reward, done, {"phaseDurations": current_phases}

    def get_state(self):
        """
        Example: queue length for each lane + current phase index
        """
        state = []
        for lane_id in self.lanes:
            # queue length or halting vehicles, as you prefer
            q_len = self.conn.lane.getLastStepHaltingNumber(lane_id)
            state.append(q_len)

        current_phase_idx = self.conn.trafficlight.getPhase(self.tls_id)
        state.append(current_phase_idx)
        return np.array(state, dtype=float)

    def calculate_reward(self):
        """
        Reward = negative of total queue (smaller queue => higher reward).
        """
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


########################################################################
# 3. DQN Agent
########################################################################

class DQNAgent:
    """
    4 discrete actions: incGreen, decGreen, incRed, decRed
    """
    def __init__(self, state_size, action_size,
                 learning_rate=0.001, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 batch_size=32, memory_size=2000):

        self.state_size     = state_size
        self.action_size    = action_size
        self.learning_rate  = learning_rate
        self.gamma          = gamma
        self.epsilon        = epsilon
        self.epsilon_min    = epsilon_min
        self.epsilon_decay  = epsilon_decay
        self.batch_size     = batch_size

        self.memory = deque(maxlen=memory_size)

        self.model        = self._build_model()
        self.target_model = self._build_model()
        self._update_target_model()

    def _build_model(self):
        # GPU usage if available
        with tf.device("/device:GPU:0"):
            model = keras.Sequential()
            model.add(keras.layers.Dense(64, input_dim=self.state_size, activation='relu'))
            model.add(keras.layers.Dense(64, activation='relu'))
            model.add(keras.layers.Dense(self.action_size, activation='linear'))
            model.compile(
                loss='mse',
                optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate)
            )
        return model

    def _update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        with tf.device("/device:GPU:0"):
            q_values = self.model.predict(state[np.newaxis, :])
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
                target = self.model.predict(state[np.newaxis, :])[0]
                if done:
                    target[action] = reward
                else:
                    t = self.target_model.predict(next_state[np.newaxis, :])[0]
                    target[action] = reward + self.gamma * np.amax(t)
            targets[i] = target

        with tf.device("/device:GPU:0"):
            self.model.fit(states, targets, epochs=1, verbose=0)

        # Epsilon-greedy decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


########################################################################
# 4. Training Function
########################################################################

def train_dqn_fixed_steps(sumo_cfg_path, tls_id, edges, lanes,
                          n_episodes=10, max_steps=3500,
                          rl_steps_per_episode=50):
    """
    Train the DQN agent with a FIXED number of RL steps (actions) per episode.
    Each action changes green/red durations, then we run the entire cycle.
    We'll track only ACTIVE vehicles in the logs, not TOT vehicles.
    """

    env = SumoEnvironment(
        sumo_cfg_path   = sumo_cfg_path,
        tls_id          = tls_id,
        edges           = edges,
        lanes           = lanes,
        reward_function = 'queue_length',
        max_steps       = max_steps
    )

    # 1) Grab the state size
    initial_state = env.reset(episode_idx=0)
    state_size = len(initial_state)

    # 2) DQN Agent
    action_size = 4  # incGreen, decGreen, incRed, decRed
    agent = DQNAgent(
        state_size   = state_size,
        action_size  = action_size
    )

    # 3) Excel logging
    workbook  = xlsxwriter.Workbook('RL_Training_Results.xlsx')
    worksheet = workbook.add_worksheet('Results')

    headers = [
        "Episode",
        "Step",
        "State",
        "tls_id",
        "edge_id",
        "lane_id",
        "Action",
        "PhaseDurations",  # [G, Y, R]
        "ActiveCars",      # <--- we'll store the total active vehicles
        "Reward"
    ]
    for col, header in enumerate(headers):
        worksheet.write(0, col, header)

    row_excel = 1

    # We'll also track total reward each episode to see progress
    episode_rewards = []

    # 4) Main Training Loop
    for e in range(n_episodes):
        state = env.reset(episode_idx=e)
        total_reward = 0

        for step_counter in range(rl_steps_per_episode):
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)

            # Store transition & train
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            agent._update_target_model()

            # This time, let's log the "ACTIVE" vehicles, i.e. how many are on the road
            # We can approximate by summing getLastStepVehicleNumber for each lane
            active_cars = 0
            for lane_id in lanes:
                # active_cars += env.conn.lane.getLastStepVehicleNumber(lane_id)
                active_cars += env.conn.lane.getLastStepHaltingNumber(lane_id)


            # We'll log for each lane
            for lane_id in lanes:
                worksheet.write(row_excel, 0, e)               # Episode
                worksheet.write(row_excel, 1, step_counter)    # RL Step
                worksheet.write(row_excel, 2, str(state.tolist()))
                worksheet.write(row_excel, 3, tls_id)
                worksheet.write(row_excel, 4, " - ".join(edges))
                worksheet.write(row_excel, 5, lane_id)
                worksheet.write(row_excel, 6, action)
                worksheet.write(row_excel, 7, str(info["phaseDurations"]))  # [G,Y,R]
                worksheet.write(row_excel, 8, active_cars)     # ActiveCars
                worksheet.write(row_excel, 9, reward)
                row_excel += 1

            state = next_state
            total_reward += reward

            if done:
                print(f"Episode {e+1}/{n_episodes} ended early at RL step {step_counter}.")
                break

        # After finishing an episode (or if done early), track total reward
        episode_rewards.append(total_reward)
        print(f"[Episode {e+1}/{n_episodes}] Total Reward: {total_reward:.2f}")

    # 5) Close Excel & environment
    workbook.close()
    env.close()

    # Optional: Print average reward across all episodes to see if it improved
    avg_reward = np.mean(episode_rewards)
    print(f"Average reward across {n_episodes} episodes: {avg_reward:.2f}")


########################################################################
# 5. MAIN SCRIPT
########################################################################

if __name__ == "__main__":
    # Optional: check GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("GPUs found:", gpus)
    else:
        print("No GPU found. Using CPU.")

    SUMO_CFG_PATH = "/content/sumofilesA/A.sumocfg"

    print("Testing SUMO config offline:")
    !sumo -c $SUMO_CFG_PATH

    # Retrieve network info
    tls_ids, edges, lanes = get_sumo_info(SUMO_CFG_PATH, label="infoConnection")
    print("Traffic Light IDs:", tls_ids)
    print("Edges:", edges)
    print("Lanes:", lanes)

    if not tls_ids:
        raise ValueError("No traffic lights found in the network.")
    my_tls_id = tls_ids[0]

    # Retrieve initial phases (just for reference)
    phase_durations = get_tls_phases(SUMO_CFG_PATH, my_tls_id, label="phaseConnection")
    print("Initial phase durations:", phase_durations)

    # Train with EXACTLY 50 RL steps per episode (unless done early)
    train_dqn_fixed_steps(
        sumo_cfg_path       = SUMO_CFG_PATH,
        tls_id              = my_tls_id,
        edges               = edges,
        lanes               = lanes,
        n_episodes          = 8,
        max_steps           = 3500,
        rl_steps_per_episode= 50
    )
