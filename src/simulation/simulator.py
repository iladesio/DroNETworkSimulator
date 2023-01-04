import math
import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.drawing import pp_draw
from src.entities.uav_entities import *
from src.routing_algorithms.deep_ql.dqn import DQN
from src.routing_algorithms.deep_ql.replay_memory import ReplayMemory, Transition
from src.routing_algorithms.net_routing import MediumDispatcher
from src.simulation.metrics import Metrics
from src.utilities import config, utilities

"""
This file contains the Simulation class. It allows to explicit all the relevant parameters of the simulation,
as default all the parameters are set to be those in the config file. For extensive experimental campains, 
you can initialize the Simulator with non default values. 
"""

LR = 1e-4
BATCH_SIZE = 128
GAMMA = 0.99
TAU = 1  # 0.005


class Simulator:

    def __init__(self,
                 len_simulation=config.SIM_DURATION,
                 time_step_duration=config.TS_DURATION,
                 seed=config.SEED,
                 n_drones=config.N_DRONES,
                 env_width=config.ENV_WIDTH,
                 env_height=config.ENV_HEIGHT,
                 drone_com_range=config.COMMUNICATION_RANGE_DRONE,
                 drone_sen_range=config.SENSING_RANGE_DRONE,
                 drone_speed=config.DRONE_SPEED,
                 drone_max_buffer_size=config.DRONE_MAX_BUFFER_SIZE,
                 drone_max_energy=config.DRONE_MAX_ENERGY,
                 drone_retransmission_delta=config.RETRANSMISSION_DELAY,
                 drone_communication_success=config.COMMUNICATION_P_SUCCESS,
                 depot_com_range=config.DEPOT_COMMUNICATION_RANGE,
                 depot_coordinates=config.DEPOT_COO,
                 event_duration=config.EVENTS_DURATION,
                 event_generation_prob=config.P_FEEL_EVENT,
                 event_generation_delay=config.D_FEEL_EVENT,
                 packets_max_ttl=config.PACKETS_MAX_TTL,
                 show_plot=config.PLOT_SIM,
                 routing_algorithm=config.ROUTING_ALGORITHM,
                 communication_error_type=config.CHANNEL_ERROR_TYPE,
                 prob_size_cell_r=config.CELL_PROB_SIZE_R,
                 simulation_name=""):
        self.cur_step = None
        self.drone_com_range = drone_com_range
        self.drone_sen_range = drone_sen_range
        self.drone_speed = drone_speed
        self.drone_max_buffer_size = drone_max_buffer_size
        self.drone_max_energy = drone_max_energy
        self.drone_retransmission_delta = drone_retransmission_delta
        self.drone_communication_success = drone_communication_success
        self.n_drones = n_drones
        self.env_width = env_width
        self.env_height = env_height
        self.depot_com_range = depot_com_range
        self.depot_coordinates = depot_coordinates
        self.len_simulation = len_simulation
        self.time_step_duration = time_step_duration
        self.seed = seed
        self.event_duration = event_duration
        self.event_max_retrasmission = math.ceil(event_duration / drone_retransmission_delta)
        self.event_generation_prob = event_generation_prob
        self.event_generation_delay = event_generation_delay
        self.packets_max_ttl = packets_max_ttl
        self.show_plot = show_plot
        self.routing_algorithm = routing_algorithm
        self.communication_error_type = communication_error_type
        self.max_connection_time = None
        self.sim_metrics = {}

        # --------------- cell for drones -------------
        self.prob_size_cell_r = prob_size_cell_r
        self.prob_size_cell = int(self.drone_com_range * self.prob_size_cell_r)
        self.cell_prob_map = defaultdict(lambda: [0, 0, 0])

        self.sim_save_file = config.SAVE_PLOT_DIR + self.__sim_name()
        self.path_to_depot = None

        # Setup vari
        # for stats
        self.metrics = Metrics(self)

        # setup network
        self.__setup_net_dispatcher()

        # Setup the simulation
        self.__set_simulation()
        self.__set_metrics()

        self.simulation_name = "out__" + str(self.seed) + "_" + str(self.n_drones) + "_" + str(self.routing_algorithm)
        self.simulation_test_dir = self.simulation_name + "/"

        self.start = time.time()
        self.event_generator = utilities.EventGenerator(self)

        if self.routing_algorithm.name == "ARDEEP_QL":
            self.n_actions = self.n_drones
            self.n_observations = self.n_drones

            self.policy_net = DQN(self.n_observations * 5, self.n_actions).to(config.DEVICE)

            # get model from the file
            if config.READ_MODEL_DICT:
                try:
                    self.policy_net.load_state_dict(torch.load(config.MODEL_STATE_DICT_PATH))
                    self.policy_net.eval()
                except RuntimeError:
                    print("ERROR! Cannot load model from file")

            self.target_net = DQN(self.n_observations * 5, self.n_actions).to(config.DEVICE)
            self.target_net.load_state_dict(self.policy_net.state_dict())

            self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
            self.memory = ReplayMemory(10000)

            self.weights_l1 = self.policy_net.layer1.weight.data.tolist()
            self.weights_l2 = self.policy_net.layer2.weight.data.tolist()
            self.weights_l3 = self.policy_net.layer3.weight.data.tolist()

            self.train_count = 0

        print("Device: ", config.DEVICE)

    def __setup_net_dispatcher(self):
        self.network_dispatcher = MediumDispatcher(self.metrics)

    def __set_metrics(self):
        """ the method sets up all the parameters in the metrics class """
        self.metrics.info_mission()

    def __set_random_generators(self):
        if self.seed is not None:
            self.rnd_network = np.random.RandomState(self.seed)
            self.rnd_routing = np.random.RandomState(self.seed)
            self.rnd_env = np.random.RandomState(self.seed)
            self.rnd_event = np.random.RandomState(self.seed)

    def __set_simulation(self):
        """ the method creates all the uav entities """

        self.__set_random_generators()

        self.path_manager = utilities.PathManager(config.PATH_FROM_JSON, config.JSONS_PATH_PREFIX, self.seed)
        self.environment = Environment(self.env_width, self.env_height, self)

        self.depot = Depot(self.depot_coordinates, self.depot_com_range, self)

        self.drones = []

        # drone 0 is the first
        for i in range(self.n_drones):
            self.drones.append(Drone(i, self.path_manager.path(i, self), self.depot, self))

        self.environment.add_drones(self.drones)
        self.environment.add_depot(self.depot)

        # Set the maximum distance between the drones and the depot
        self.max_dist_drone_depot = utilities.euclidean_distance(self.depot.coords, (self.env_width, self.env_height))

        if self.show_plot or config.SAVE_PLOT:
            self.draw_manager = pp_draw.PathPlanningDrawer(self.environment, self, borders=True)

    def __sim_name(self):
        """
            return the identification name for
            the current simulation. It is useful to print
            the simulation progress
        """
        return "sim_seed" + str(self.seed) + "drones" + str(self.n_drones) + "_step"

    def __plot(self, cur_step):
        """ plot the simulation """
        if cur_step % config.SKIP_SIM_STEP != 0:
            return

        # delay draw
        if config.WAIT_SIM_STEP > 0:
            time.sleep(config.WAIT_SIM_STEP)

        # drones plot
        for drone in self.drones:
            self.draw_manager.draw_drone(drone, cur_step)

        # depot plot
        self.draw_manager.draw_depot(self.depot)

        # events
        for event in self.environment.active_events:
            self.draw_manager.draw_event(event)

        # Draw simulation info
        self.draw_manager.draw_simulation_info(cur_step=cur_step, max_steps=self.len_simulation)

        # rendering
        self.draw_manager.update(show=self.show_plot, save=config.SAVE_PLOT,
                                 filename=self.sim_save_file + str(cur_step) + ".png")

    def increase_meetings_probs(self, drones, cur_step):
        """ Increases the probabilities of meeting someone. """
        cells = set()
        for drone in drones:
            coords = drone.coords
            cell_index = utilities.TraversedCells.coord_to_cell(size_cell=self.prob_size_cell,
                                                                width_area=self.env_width,
                                                                x_pos=coords[0],  # e.g. 1500
                                                                y_pos=coords[1])  # e.g. 500
            cells.add(int(cell_index[0]))

        for cell, cell_center in utilities.TraversedCells.all_centers(self.env_width, self.env_height,
                                                                      self.prob_size_cell):

            index_cell = int(cell[0])
            old_vals = self.cell_prob_map[index_cell]

            if index_cell in cells:
                old_vals[0] += 1

            old_vals[1] = cur_step + 1
            old_vals[2] = old_vals[0] / max(1, old_vals[1])
            self.cell_prob_map[index_cell] = old_vals

    def run(self):
        """
        Simulator main function
        @return: None
        """

        """ ---- initialize simulator variabiles ---- """
        self.max_connection_time = utilities.get_max_connection_time(self.drones)

        for drone in self.drones:
            self.metrics.rewards_actions[drone.identifier] = {}

            for action in self.drones:
                self.metrics.rewards_actions[drone.identifier][action.identifier] = []
        """ ----------------------------------------- """

        for cur_step in tqdm(range(self.len_simulation)):

            self.cur_step = cur_step
            # check for new events and remove the expired ones from the environment
            # self.environment.update_events(cur_step)
            # sense the area and move drones and sense the area
            self.network_dispatcher.run_medium(cur_step)

            # generates events
            # sense the events
            self.event_generator.handle_events_generation(cur_step, self.drones)

            if config.SAVE_CONNECTION_TIME_DATA and self.routing_algorithm.name == "ARDEEP_QL":
                for drone in self.drones:
                    self.get_connection_time_data(cur_step, drone)

            for drone in self.drones:
                # 1. update expired packets on drone buffers
                # 2. try routing packets vs other drones or depot
                # 3. actually move the drone towards next waypoint or depot

                drone.update_packets(cur_step)
                drone.routing(self.drones, self.depot, cur_step)
                drone.move(self.time_step_duration)

            """ ----------------- update next_state ----------------- """
            # todo da rivedere delta
            if self.routing_algorithm.name == "ARDEEP_QL" and cur_step % config.CALCULATE_NEXT_STATE_DELTA == 0:
                for drone in self.drones:
                    list_neighbors = [d[0] for d in drone.get_neighbours()]

                    # get next state
                    drone.routing_algorithm.update_next_state(list_neighbors, self.cur_step + 1)

            # in case we need probability map
            if config.ENABLE_PROBABILITIES:
                self.increase_meetings_probs(self.drones, cur_step)

            if self.show_plot or config.SAVE_PLOT:
                self.__plot(cur_step)

            # train model
            if self.routing_algorithm.name == "ARDEEP_QL" and config.TRAIN_MODEL:
                self.optimize_model()

        if config.SAVE_CONNECTION_TIME_DATA:
            utilities.save_connection_time_data(self.drones)

        if config.DEBUG:
            print("End of simulation, sim time: " + str(
                (cur_step + 1) * self.time_step_duration) + " sec, #iteration: " + str(cur_step + 1))

    def get_connection_time_data(self, cur_step, drone):

        error_range = 5

        curr_open_connection_drones = set()
        prev_open_connection_drones = set()

        # get the list of previous opened connection
        for neighbour in drone.neighbor_connection_time.keys():
            if drone.neighbor_connection_time[neighbour] and \
                    drone.neighbor_connection_time[neighbour][-1][1] is None:
                prev_open_connection_drones.add(neighbour)

        neighbours = [n[0] for n in drone.get_neighbours()]

        for neigh in neighbours:

            id_drone = neigh.identifier
            conn_time_history = drone.neighbor_connection_time[id_drone]

            # if we meet the current neighbor for the first time or
            # there isn't an opened connection, we create it
            if not conn_time_history or (conn_time_history and conn_time_history[-1][1] is not None):
                conn_time_history.append([cur_step - error_range, None])

            # we save in a set all current opened connection
            curr_open_connection_drones.add(id_drone)

        # close previous opened connection that doesn't appear in the current neighbours list
        for neighbour in prev_open_connection_drones:
            if neighbour not in curr_open_connection_drones:
                drone.neighbor_connection_time[neighbour][-1][1] = cur_step + error_range

    """ TRAINING MODEL METHOD """

    def optimize_model(self):

        # Perform one step of the optimization (on the policy network)
        if len(self.memory) < BATCH_SIZE:
            return

        self.train_count += 1

        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        # todo check if we can remove non_final_mask
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                      device=config.DEVICE,
                                      dtype=torch.bool)

        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=config.DEVICE)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ) θ′
        if self.cur_step % config.TRAINING_DELTA == 0:
            target_net_state_dict = self.target_net.state_dict()
            policy_net_state_dict = self.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + \
                                             target_net_state_dict[key] * (1 - TAU)

            self.target_net.load_state_dict(target_net_state_dict)

        # save the model
        if config.SAVE_MODEL_DICT:
            torch.save(self.target_net.state_dict(), config.MODEL_STATE_DICT_PATH)

    def close(self):
        """ do some stuff at the end of simulation"""
        print("Closing simulation")
        print("Len memory: ", len(self.memory))
        print("Train count: ", self.train_count)

        self.print_metrics(plot_id="final")
        # make sure to have output directory in the project
        # self.save_metrics(config.ROOT_EVALUATION_DATA + self.simulation_name)

        # weights_l1_final = self.policy_net.layer1.weight.data.tolist()
        # weights_l2_final = self.policy_net.layer2.weight.data.tolist()
        # weights_l3_final = self.policy_net.layer3.weight.data.tolist()
        #
        # weights_diff = []
        #
        # for r in range(len(weights_l1_final)):
        #     row = []
        #     for i in range(len(weights_l1_final[r])):
        #         diff = weights_l1_final[r][i] - self.weights_l1[r][i]
        #         if diff > 0.1:
        #             print("sium " + diff)
        #         row.append(diff)
        #
        #     weights_diff.append(row)
        #
        # print(weights_diff)

    def get_metrics(self):
        self.sim_metrics.update(self.metrics.get_metrics())
        self.sim_metrics["len memory"] = len(self.memory)

        return self.sim_metrics

    def print_metrics(self, plot_id="final"):
        """ add signature """
        self.metrics.print_overall_stats()

    def save_metrics(self, filename_path, save_pickle=False):
        """ add signature """
        self.metrics.save_as_json(filename_path + ".json")
        if save_pickle:
            self.metrics.save(filename_path + ".pickle")
