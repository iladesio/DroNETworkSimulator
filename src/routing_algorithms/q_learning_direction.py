import random

import numpy as np

from src.routing_algorithms.BASE_routing import BASE_routing
from src.routing_algorithms.georouting import GeoRouting
from src.utilities import utilities as util

optimal_init_value = 20


class QLearningRoutingDirection(BASE_routing):

    def __init__(self, drone, simulator):
        BASE_routing.__init__(self, drone=drone, simulator=simulator)
        self.taken_actions = {}

        """     Representation of the states                        Optimistic value for 
                (direction, cell_index)                             each state
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
            (1, 0): [0, optimal_init_value],                     # ?
            (2, 0): [0, optimal_init_value],    # Move
            (3, 0): [0, optimal_init_value],    # Move
            (4, 0): [optimal_init_value, 0],    # Keep

            (1, 1): [0, optimal_init_value],    # Move
            (2, 1): [0, optimal_init_value],                     # ?
            (3, 1): [optimal_init_value, 0],    # Keep
            (4, 1): [0, optimal_init_value],    # Move

            (1, 2): [0, optimal_init_value],    # Move
            (2, 2): [0, optimal_init_value],    # Move
            (3, 2): [0, optimal_init_value],                     # ?
            (4, 2): [optimal_init_value, 0],    # Keep

            (1, 3): [0, optimal_init_value],    # Move
            (2, 3): [0, optimal_init_value],    # Move
            (3, 3): [optimal_init_value, 0],    # Keep
            (4, 3): [0, optimal_init_value],                     # ?
        }

        # Learning Rate
        self.alpha = 0.6  # ToModify

        # Discount Factor
        self.gamma = 0.8  # ToModify

        # Epsilon
        self.epsilon = 0.95  # ToModify

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
        #   1 if the packets has been delivered to the depot

        # Be aware, due to network errors we can give the same event to multiple drones and receive multiple
        # feedback for the same packet!! # todo gestire questo caso (?)

        if id_event in self.taken_actions:
            # BE AWARE, IMPLEMENT YOUR CODE WITHIN THIS IF CONDITION OTHERWISE IT WON'T WORK!
            # TIPS: implement here the q-table updating process
            # Drone id and Taken actions

            state, action, next_drone, current_waypoint = self.taken_actions[id_event]

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

                current_cell = util.TraversedCells.coord_to_cell(size_cell=self.simulator.prob_size_cell,
                                                                 width_area=self.simulator.env_width,
                                                                 x_pos=p1[0],  # e.g. 1500
                                                                 y_pos=p1[1])[0]  # e.g. 500

                current_direction = util.map_angle_to_state(util.get_angle_degree(p1, p2))
                next_state = (current_direction, int(current_cell))

            else:
                next_state = state

            # reward or update using the old state and the selected action at that time
            # do something or train the model (?)
            if outcome == 1:
                reward = (2000 - delay) / 20
            else:
                reward = -20

            gamma = self.simulator.cur_step / self.simulator.len_simulation

            # receive the reward for moving to the new state, and calculate the temporal difference
            old_q_value = self.q_table[old_state][old_action]
            temporal_difference = reward + gamma * (max(self.q_table[next_state])) - old_q_value

            # update the Q-value for the previous state and action pair
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

        # event already present in taken_action -> action already decided
        # Todo: perché dovrei riprendere la stessa scelta se sto in uno stato diverso?
        if packet.event_ref.identifier in self.taken_actions:
            state, action, next_drone, current_waypoint = self.taken_actions[packet.event_ref.identifier]
            # keep
            if action == 0:
                return None
            # move
            else:
                return next_drone

        direction = util.map_angle_to_state(util.get_angle_degree(self.drone.coords, self.drone.next_target()))

        cell_index = util.TraversedCells.coord_to_cell(size_cell=self.simulator.prob_size_cell,
                                                       width_area=self.simulator.env_width,
                                                       x_pos=self.drone.coords[0],  # e.g. 1500
                                                       y_pos=self.drone.coords[1])[0]  # e.g. 500

        state = (direction, int(cell_index))

        # Strategy: Epsilon-Greedy
        # Decide if the agent should explore or exploit using epsilon
        p = random.random()

        if p > self.epsilon:
            should_explore = True
        else:
            should_explore = False

        if not should_explore:
            # Exploit - choose the best action
            if self.drone.current_waypoint == 0:
                action = random.randint(0, 1)
            else:
                action = np.argmax(self.q_table[state])

        else:
            # Explore
            # choose a random action
            action = random.randint(0, 1)

        if len(neighbors) == 0:
            action = 0

        # keep the packet
        if action == 0:
            next_drone = self.drone
        # send the packet
        else:
            # next_drone = GeoRouting.relay_selection(self, opt_neighbors, packet)
            # next_drone = self.simulator.rnd_routing.choice(neighbors)

            next_drone = neighbors[0]
            is_some_neighbor_going_to_depot = False
            for nb in neighbors:
                if util.distance_to_line(self.simulator.depot_coordinates, nb.coords, nb.next_target()) <= 200:
                    next_drone = nb
                    is_some_neighbor_going_to_depot = True

            if not is_some_neighbor_going_to_depot:
                next_drone = GeoRouting.relay_selection(self, opt_neighbors, packet)

        # next_drone could be None after geo routing selection
        if next_drone is None:
            action = 0
            next_drone = self.drone

        current_waypoint = self.drone.current_waypoint

        # Store current action
        self.taken_actions[packet.event_ref.identifier] = (state, action, next_drone, current_waypoint)

        # action = 0 => self
        # action = 1 => next_drone neighbour
        return None if action == 0 else next_drone
