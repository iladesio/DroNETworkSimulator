import random

import numpy as np

from src.routing_algorithms.BASE_routing import BASE_routing
from src.routing_algorithms.georouting import GeoRouting
from src.utilities import utilities as util

optimistic_init_value = 10


class QLearningRoutingDirection(BASE_routing):

    def __init__(self, drone, simulator):
        BASE_routing.__init__(self, drone=drone, simulator=simulator)
        self.taken_actions = {}

        """     Representation of the states                        Optimistic value for 
                (direction, cell_index)                             each state (M = Move, K = Keep)
        +------------------+------------------+                 +----------+----------+             
        |  (2,2) | (1,2)   |   (2,3) | (1,3)  |                 |  M | M   |   M | M  |             
        |  ------+------   |   ------+------  |                 |  --+--   |   --+--  |   
        |  (3,2) | (4,2)   |   (3,3) | (4,3)  |                 |  M | K   |   K | M  |
        |------------------+------------------|                 |----------+----------|
        |  (2,0) | (1,0)   |   (2,1) | (1,1)  |                 |  M | M   |   M | M  |
        |  ------+------   |   ------+------  |                 |  --+--   |   --+--  |
        |  (3,0) | (4,0)   |   (3,1) | (4,1)  |                 |  M | K   |   K | M  |
        +------------------+------------------+                 +----------+----------+
        """
        self.q_table = {
            (1, 0): [0, optimistic_init_value],  # Move
            (2, 0): [0, optimistic_init_value],  # Move
            (3, 0): [0, optimistic_init_value],  # Move
            (4, 0): [optimistic_init_value, 0],  # Keep

            (1, 1): [0, optimistic_init_value],  # Move
            (2, 1): [0, optimistic_init_value],  # Move
            (3, 1): [optimistic_init_value, 0],  # Keep
            (4, 1): [0, optimistic_init_value],  # Move

            (1, 2): [0, optimistic_init_value],  # Move
            (2, 2): [0, optimistic_init_value],  # Move
            (3, 2): [0, optimistic_init_value],  # Move
            (4, 2): [optimistic_init_value, 0],  # Keep

            (1, 3): [0, optimistic_init_value],  # Move
            (2, 3): [0, optimistic_init_value],  # Move
            (3, 3): [optimistic_init_value, 0],  # Keep
            (4, 3): [0, optimistic_init_value],  # Move
        }

        # Learning Rate
        self.alpha = 0.0  # calculate at each timestep

        # Discount Factor
        self.gamma = 0.0  # calculate at each timestep

        # Epsilon
        self.epsilon = 0.95

        # use this to have always the same simulation
        random.seed(self.simulator.seed)

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
            # BE AWARE, IMPLEMENT YOUR CODE WITHIN THIS IF CONDITION OTHERWISE IT WON'T WORK!
            # TIPS: implement here the q-table updating process
            # Drone id and Taken actions

            state, action, current_waypoint = self.taken_actions[id_event]

            old_state = state
            old_action = action

            # current_waypoint = waypoint in which action was taken - postincrement - starts from 0
            # self.drone.waypoint_history is a list of waypoint coords without the current drone

            if self.drone.current_waypoint != current_waypoint:
                sub_history = self.drone.waypoint_history[current_waypoint:]

                p1 = sub_history[0]

                sub_history = self.drone.waypoint_history[current_waypoint + 1:]

                if len(sub_history) == 0:
                    p2 = self.drone.next_target()
                else:
                    p2 = sub_history[0]

                current_cell = util.TraversedCells.coord_to_cell(size_cell=self.simulator.env_width / 4,
                                                                 width_area=self.simulator.env_width,
                                                                 x_pos=p1[0],
                                                                 y_pos=p1[1])[0]

                current_direction = util.map_angle_to_state(util.get_angle_degree(p1, p2))
                next_state = (current_direction, int(current_cell))

            else:
                next_state = state

            # reward or update using the old state and the selected action at that time
            # do something or train the model (?)
            if outcome == 1:
                reward = (2000 - delay) / 50  # to a maximum of 40
            else:
                reward = -20

            if self.simulator.cur_step / self.simulator.len_simulation < 0.25:
                self.alpha = 0.8
            elif self.simulator.cur_step / self.simulator.len_simulation < 0.50:
                self.alpha = 0.6
            elif self.simulator.cur_step / self.simulator.len_simulation < 0.75:
                self.alpha = 0.4
            else:
                self.alpha = 0.2

            self.gamma = self.simulator.cur_step / self.simulator.len_simulation

            # receive the reward for moving to the new state, and calculate the temporal difference
            old_q_value = self.q_table[old_state][old_action]
            temporal_difference = reward + self.gamma * (max(self.q_table[next_state])) - old_q_value

            # Bellman equation: update the Q-value for the previous state and action pair
            new_q_value = old_q_value + (self.alpha * temporal_difference)
            self.q_table[old_state][old_action] = new_q_value

            # remove the entry, the action has received the feedback
            del self.taken_actions[id_event]

    def relay_selection(self, opt_neighbors: list, packet):

        """
        This function returns the best relay to send packets.
        @param packet:
        @param opt_neighbors: a list of tuple (hello_packet, source_drone)
        @return: The best drone to use as relay
        """

        neighbors = [n[1] for n in opt_neighbors]

        direction = util.map_angle_to_state(util.get_angle_degree(self.drone.coords, self.drone.next_target()))

        cell_index = util.TraversedCells.coord_to_cell(size_cell=self.simulator.env_width / 4,
                                                       # size_cell=self.simulator.prob_size_cell
                                                       width_area=self.simulator.env_width,
                                                       x_pos=self.drone.coords[0],  # e.g. 1500
                                                       y_pos=self.drone.coords[1])[0]  # e.g. 500

        state = (direction, int(cell_index))

        # Strategy: Epsilon-Greedy
        # Decide if the agent should explore or exploit using epsilon

        p = random.uniform(0, 1)

        if p > self.epsilon:
            should_explore = True
        else:
            should_explore = False

        next_drone = None

        range = self.drone.depot.communication_range

        if len(neighbors) == 0 or util.euclidean_distance(self.simulator.depot_coordinates,
                                                          self.drone.next_target()) <= range:
            action = 0  # keep

        else:
            # if one of your neighbors goes to the depot, give him all the packets
            is_some_neighbor_going_to_depot = False
            for nb in neighbors:
                if ((util.distance_to_line(self.simulator.depot_coordinates, nb.coords, nb.next_target()) <= range and
                     (nb.coords[1] <= range or nb.next_target()[1] <= range))
                        or util.euclidean_distance(self.simulator.depot_coordinates, nb.next_target()) <= range):
                    next_drone = nb
                    action = 1
                    is_some_neighbor_going_to_depot = True

            if not is_some_neighbor_going_to_depot:
                if not should_explore:
                    # Exploit - choose the best action
                    action = np.argmax(self.q_table[state])

                else:
                    # Explore
                    # choose a random action
                    action = random.randint(0, 1)

                # Send the packet
                if action == 1:
                    next_drone = GeoRouting.relay_selection(self, opt_neighbors, packet)
                    if next_drone is None:
                        action = 0

        current_waypoint = self.drone.current_waypoint

        # Store your current action --- you can add some stuff if needed to take a reward later
        self.taken_actions[packet.event_ref.identifier] = (state, action, current_waypoint)

        return next_drone
