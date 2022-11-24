from src.routing_algorithms.BASE_routing import BASE_routing
from src.utilities import utilities as util

import numpy as np

from src.routing_algorithms import georouting as geo


class QLearningRouting(BASE_routing):

    def __init__(self, drone, simulator):
        BASE_routing.__init__(self, drone=drone, simulator=simulator)
        self.taken_actions = {}  # id event : (old_state, old_action)  -->  id event : (old_state, old_action, new_state)

        self.q_table = {}  # state : list of drones
        """
        State   |                Actions
        --------+-----------+-----------+-----+------------
        cell_1  |  drone_1  |  drone_2  | ... |  drone_n  |
        --------+-----------+-----------+-----+------------
          ...   |  drone_1  |  drone_2  | ... |  drone_n  |
        --------+-----------+-----------+-----+------------
        cell_m  |  drone_1  |  drone_2  | ... |  drone_n  |
        """


        # Learning Rate
        self.alpha = 0.6  # ToModify

        # Discount Factor
        self.gamma = 0.8  # ToModify

        # epsilon
        self.epsilon = 0.9  # ToModify

        self._rng = np.random.default_rng()

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
        # feedback for the same packet!!

        if id_event in self.taken_actions:
            # BE AWARE, IMPLEMENT YOUR CODE WITHIN THIS IF CONDITION OTHERWISE IT WON'T WORK!
            # TIPS: implement here the q-table updating process

            # Drone id and Taken actions
            """
            if self.simulator.cur_step < 4000:
                print(
                    f"\nIdentifier: {self.drone.identifier}, Taken Actions: {self.taken_actions}, Time Step: {self.simulator.cur_step}")

                # feedback from the environment
                print("Feedback: ", drone, id_event, delay, outcome)
            """
            # TODO: write your code here

            # state, action = self.taken_actions[id_event]

            state, action, next_state = self.taken_actions[id_event]

            if action is None:
                action = self.drone.identifier
            else:
                action = action.identifier

            # remove the entry, the action has received the feedback
            del self.taken_actions[id_event]

            # reward or update using the old state and the selected action at that time
            # do something or train the model (?)

            # ToDo: Reward function
            # reward = 0

            if outcome == 1:
                reward = 2000/(delay/50+1)-44
            else:
                reward = -70


            # receive the reward for moving to the new state, and calculate the temporal difference
            q_old_value = self.q_table[state][action]
            temporal_difference = reward + self.gamma * (max(self.q_table[next_state])) - q_old_value

            # update the Q-value for the previous state and action pair
            new_q_value = q_old_value + (self.alpha * temporal_difference)
            self.q_table[state][action] = new_q_value

            print("\nQ_table: ")

            for row in self.q_table:
                print(row, ": ", self.q_table[row])

    def relay_selection(self, opt_neighbors: list, packet):
        """
        This function returns the best relay to send packets.
        @param packet:
        @param opt_neighbors: a list of tuple (hello_packet, source_drone)
        @return: The best drone to use as relay
        """
        # TODO: Implement your code HERE

        # Only if you need!
        cell_index = util.TraversedCells.coord_to_cell(size_cell=self.simulator.prob_size_cell,
                                                       width_area=self.simulator.env_width,
                                                       x_pos=self.drone.coords[0],  # e.g. 1500
                                                       y_pos=self.drone.coords[1])[0]  # e.g. 500
        # print(cell_index)
        state, action = cell_index, None

        # print(self.drone.identifier)

        if cell_index not in self.q_table:
            self.q_table[cell_index] = [0 for i in range(self.simulator.n_drones)]# + 1)]

        neighbors = [n[1] for n in opt_neighbors]

        # Strategy: Epsilon-Greedy
        # Decide if the agent should explore or exploit using epsilon
        p = np.random.random()
        if p > self.epsilon:
            should_explore = True
        else:
            should_explore = False

        if not should_explore:
            # Exploit #
            # chose the best action
            if len(neighbors) == 0:
                return None
            else:
                max_score = float('-inf')
                for n in neighbors:
                    if self.q_table[cell_index][n.identifier] > max_score:
                        action = n
                        max_score = self.q_table[cell_index][n.identifier]
        else:
            # Explore #
            # chose a random action
            if len(neighbors) == 0:
                action = None
            else:
                action = self.simulator.rnd_routing.choice(neighbors)
                # action = geo.GeoRouting.relay_selection(self, opt_neighbors, packet)

        # keep the packet
        if action is None:
            next = self.drone.next_target()
        # send the packet
        else:
            next = action.next_target()

        next_cell_index = util.TraversedCells.coord_to_cell(size_cell=self.simulator.prob_size_cell,
                                                            width_area=self.simulator.env_width,
                                                            x_pos=next[0],  # e.g. 1500
                                                            y_pos=next[1])[0]  # e.g. 500

        next_state = next_cell_index

        """
        if self.simulator.cur_step < 300:
            print("\nTime: ", self.simulator.cur_step)
            print("Holder drone: ", self.drone)
            print("Packet: ", self.drone.all_packets())
            print("Current_cell: ", cell_index)
            print("Coords: ", self.drone.coords)
            #print("Dist_depot: ", util.euclidean_distance(self.drone.coords, self.drone.depot.coords))
            print("Neighbors: ", neighbors)
            #print("Explore: ", should_explore)
            print("Q: ", self.q_table)
            print("Action: ", action)
            print("Next_cell: ", next_cell_index)
            print("Next_target: ", self.drone.next_target())
        """

        # Store your current action --- you can add some stuff if needed to take a reward later
        self.taken_actions[packet.event_ref.identifier] = (state, action, next_state)

        if next_cell_index not in self.q_table:
            self.q_table[next_cell_index] = [0 for i in range(self.simulator.n_drones)]# + 1)]

        return action

        """
        Doubts:
        - come fare Q-table?
        - qual è il next state?
        - se tengo il pacchetto devo calcolare la reward?
        - è normale che un pacchetto venga scambiato tra 2 droni ad ogni time_stamp?
        - ...
        
        - Utility reward function:
            - delay
            - buffer size
            - energia rimanente
        """
