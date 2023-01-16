import math
import random

import numpy as np
import torch
import torch.nn.functional as f

from src.routing_algorithms.BASE_routing import BASE_routing
from src.routing_algorithms.georouting import GeoRouting
from src.utilities import utilities as util, config

# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000


class ARDeepLearningRouting(BASE_routing):
    """
        -------- state definition --------

        C ui, bj = (ct ui,bj,         expected connection time of the link
                      PER ui, bj,     Packet Error Ratio of the link
                      e bj            remaining energy of neighbor bj
                      d bj, des,      distance between neighbor bj and destination des
                      d min)          minimum distance between a two hop neighbor bk and des

    """

    def __init__(self, drone, simulator):
        BASE_routing.__init__(self, drone=drone, simulator=simulator)

        # use this to have always the same simulation
        random.seed(self.simulator.seed)

        self.omega = 0.3  # 0 < w < 1 used to adjust the importance of reliable distance Dij in reward function

        self.R_max = 1

        self.connection_time_min = 6  # 1 sec

        # dictionary to store (cur_state, chose_action identifier, next_state) for each packet
        # key: event identifier
        # next_state is set to None the first time, and it'll be set after a delta time in the main simulator cycle
        self.taken_actions = {}

        # dictionary to store metadata about taken_action
        # key: event identifier
        # value: (drone_coords, action_coords, is_local_minimum) data saved at the moment
        #         in which it has chosen the action
        self.taken_action_meta = {}

    def select_action(self, state, opt_neighbors):
        # Strategy: Epsilon-Greedy
        # Decide if the agent should explore or exploit using epsilon
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.simulator.cur_step / EPS_DECAY)

        if sample > eps_threshold:
            # Exploit - choose the best action
            with torch.no_grad():
                # get action from the Q-NN, giving current state
                results = f.softmax(self.simulator.policy_net(state), dim=1)

                return self.simulator.drones[torch.argmax(results).item()]
        else:
            # Explore - choose a drone based on georouting
            geo_relay = GeoRouting.relay_selection(self, opt_neighbors, None)
            return self.drone if geo_relay is None else geo_relay

    def get_current_state(self, list_drones, step):
        state = {}

        connection_time = 0

        for neighbor in list_drones:
            # expected connection time of the link
            if config.USE_CONNECTION_TIME_DATA and str(neighbor.identifier) in self.drone.nb_connection_time.keys():
                is_value_from_interval = False
                for c in self.drone.nb_connection_time[str(neighbor.identifier)]:
                    if c[0] <= step <= c[1]:
                        connection_time = c[1] - step
                        is_value_from_interval = True
                        break

                # handle the case of no data is found
                if connection_time == 0. and not is_value_from_interval:
                    connection_time = int(random.uniform(0, self.simulator.max_connection_time))
            else:
                # value generated randomly
                connection_time = int(random.uniform(0, self.simulator.max_connection_time))

            # Packet Error Ratio of the link - generated randomly between 0 and 0.2
            packet_error_ratio = random.uniform(0, 0.2)

            # remaining energy of neighbor bj
            remaining_energy = neighbor.residual_energy

            # distance between neighbor bj and destination des
            dist_bj_destination = util.euclidean_distance(neighbor.coords, self.simulator.depot_coordinates)

            # minimum distance between a two hop neighbor bk and des
            nn = neighbor.get_neighbours()
            min_distance_bk_des = 9999999
            for n in nn:
                # exclude current drone from the two hop neighbors list of its neighbors
                if n[0].identifier is not self.drone.identifier:
                    distance_n_depot = util.euclidean_distance(n[0].coords, self.drone.depot.coords)
                    if distance_n_depot < min_distance_bk_des:
                        min_distance_bk_des = distance_n_depot

            # C ui, bj = (ct ui,bj,         expected connection time of the link
            #               PER ui, bj,     Packet Error Ratio of the link
            #               e_bj            remaining energy of neighbor bj
            #               d_bj, des,      distance between neighbor bj and destination des
            #               d_min)          minimum distance between a two hop neighbor bk and des

            # Normalization in range [0, 1] of all the elements of the state
            connection_time = connection_time / self.simulator.max_connection_time
            remaining_energy = remaining_energy / self.simulator.drone_max_energy
            dist_ui_destination = util.euclidean_distance(self.drone.coords, self.drone.depot.coords)

            dist_bj_destination = 1 if dist_ui_destination == 0 else \
                np.minimum(dist_bj_destination / dist_ui_destination, 1)

            min_distance_bk_des = 1 if dist_ui_destination == 0 else \
                np.minimum(min_distance_bk_des / dist_ui_destination, 1)

            state[neighbor.identifier] = [
                connection_time,
                packet_error_ratio,
                remaining_energy,
                dist_bj_destination,
                min_distance_bk_des
            ]

        """ STATE OF CURRENT DRONE """
        min_distance_bk_des = 9999999
        for n in list_drones:
            if n.identifier is not self.drone.identifier:
                distance_n_depot = util.euclidean_distance(n.coords, self.simulator.depot_coordinates)
                if distance_n_depot < min_distance_bk_des:
                    min_distance_bk_des = distance_n_depot

        connection_time = 1
        packet_error_ratio = 0

        remaining_energy = self.drone.residual_energy / self.simulator.drone_max_energy
        dist_ui_destination = util.euclidean_distance(self.drone.coords, self.simulator.depot_coordinates)
        dist_bj_destination = 1

        min_distance_bk_des = 1 if dist_ui_destination == 0 else \
            np.minimum(min_distance_bk_des / dist_ui_destination, 1)

        # build the tuple with each state starting from the complete list of drones
        # i.e. [0, (...state...), 0, 0, (...state...) ]
        complete_state = [state[drone.identifier] if drone in list_drones else [0, 0, 0, 0, 0] for drone in
                          self.simulator.drones]

        # setting current drone link with itself in the state
        complete_state[self.drone.identifier] = [
            connection_time,
            packet_error_ratio,
            remaining_energy,
            dist_bj_destination,
            min_distance_bk_des
        ]

        return complete_state

    def update_next_state(self, list_neighbors, cur_step):

        next_state = self.get_current_state(list_neighbors, cur_step)

        for event_id in self.taken_actions.keys():
            if self.taken_actions[event_id][2] is None:
                prev_state, chosen_action, ns = self.taken_actions[event_id]
                self.taken_actions[event_id] = (prev_state, chosen_action, next_state)

    def feedback(self, drone, id_event, delay, outcome):
        """
        Feedback returned when the packet arrives at the depot or
        Expire. This function have to be implemented in RL-based protocols ONLY
        @param drone: The drone that holds the packet
        @param id_event: The Event id
        @param delay: packet delay
        @param outcome: -1 or 1 (read below)
        @return:
        """
        # outcome can be:
        #   -1 if the packet/event expired;
        #    1 if the packets has been delivered to the depot
        # Be aware, due to network errors we can give the same event to multiple drones and receive multiple
        # feedback for the same packet!!

        if id_event in self.taken_actions:

            # update next_state if it is None
            if self.taken_actions[id_event][2] is None:
                list_neighbors = [d[0] for d in self.drone.get_neighbours()]
                self.update_next_state(list_neighbors, self.simulator.cur_step)

            state, action, next_state = self.taken_actions[id_event]
            drone_coords_prev, action_coords_prev, is_local_minimum = self.taken_action_meta[id_event]

            chosen_state = state[action]

            """ ------------- CALCULATE REWARD ------------- """

            # packet arrives to the depot
            if outcome == 1:
                # when neighbor bj is the destination
                if action == drone.identifier:
                    reward = self.R_max

                # when bj is a relay in the src-dest path
                else:
                    dist_ui_destination = util.euclidean_distance(drone_coords_prev, self.simulator.depot_coordinates)
                    dist_bj_destination = util.euclidean_distance(action_coords_prev, self.simulator.depot_coordinates)
                    connection_time = chosen_state[0] * self.simulator.max_connection_time  # denormalize value
                    packet_error_ratio = chosen_state[1]
                    remaining_energy = chosen_state[2]

                    beta = 1 if connection_time >= self.connection_time_min else 0

                    d_ui_bj = 1 if dist_bj_destination == 0 else \
                        dist_ui_destination / dist_bj_destination * (1 - packet_error_ratio) * beta

                    reward = self.omega * d_ui_bj + (1 - self.omega) * remaining_energy

                    # bound the reward in (0, Rmax]
                    reward = np.minimum(reward, self.R_max)

            # when neighbor bj is the local minimum
            # (all neighbors of node ui are further away from the destination than node ui)
            elif is_local_minimum is True:
                reward = - self.R_max

            # negative reward if packet expires
            else:
                dist_ui_destination = util.euclidean_distance(drone_coords_prev, self.simulator.depot_coordinates)
                dist_bj_destination = util.euclidean_distance(action_coords_prev, self.simulator.depot_coordinates)
                connection_time = chosen_state[0] * self.simulator.max_connection_time  # denormalize value
                packet_error_ratio = chosen_state[1]
                remaining_energy = chosen_state[2]

                beta = 1 if connection_time >= self.connection_time_min else 0

                d_ui_bj = 1 if dist_bj_destination == 0 else \
                    dist_ui_destination / dist_bj_destination * (1 - packet_error_ratio) * beta

                reward = - (self.R_max - (self.omega * d_ui_bj + (1 - self.omega) * remaining_energy))

                # bound the reward in [-Rmax, 0)
                reward = np.maximum(reward, -self.R_max)

            """ update metrics """
            self.simulator.metrics.rewards_actions[self.drone.identifier][action].append(reward)

            """ update tot_reward metrics """
            tot_reward = self.simulator.reward_trend[-1][1] if self.simulator.reward_trend else 0
            self.simulator.reward_trend.append([self.simulator.cur_step, tot_reward + reward])

            """ save sample in experience ReplayMemory """
            state = torch.Tensor(state).to(config.DEVICE)
            next_state = torch.Tensor(next_state).to(config.DEVICE)

            # 5 are the elements of C ui, bj
            state = torch.reshape(state, (1, self.simulator.n_observations * 5))
            next_state = torch.reshape(next_state, (1, self.simulator.n_observations * 5))

            self.simulator.memory.push(state,
                                       torch.tensor([[action]], dtype=torch.int64, device=config.DEVICE),
                                       next_state, torch.Tensor([reward]).to(config.DEVICE))

            """ remove the entry, the action has received the feedback """
            del self.taken_actions[id_event]

    def relay_selection(self, opt_neighbors: list, packet):
        """
        This function returns the best relay to send packets.
        @param packet:
        @param opt_neighbors: a list of tuple (hello_packet, source_drone)
        @return: The best drone to use as relay
        """

        list_neighbors = [n[1] for n in opt_neighbors]

        state = self.get_current_state(list_neighbors, self.simulator.cur_step)

        state_tensor = torch.Tensor(state).to(config.DEVICE)
        state_tensor = torch.reshape(state_tensor,
                                     (1, self.simulator.n_observations * 5))  # 5 are the elements of C ui, bj

        chosen_action = self.select_action(state_tensor, opt_neighbors)

        """ check if chosen action is a local minimum """
        dist_ui_des = util.euclidean_distance(chosen_action.coords, self.simulator.depot_coordinates)

        is_local_minimum = True
        nn = chosen_action.get_neighbours()

        for n in nn:
            dist_n_des = util.euclidean_distance(n[0].coords, self.simulator.depot_coordinates)
            if dist_ui_des > dist_n_des:
                is_local_minimum = False

        # store in taken actions
        self.taken_actions[packet.event_ref.identifier] = (state, chosen_action.identifier, None)
        self.taken_action_meta[packet.event_ref.identifier] = (
            self.drone.coords, chosen_action.coords, is_local_minimum)

        return None if chosen_action.identifier == self.drone.identifier else chosen_action
