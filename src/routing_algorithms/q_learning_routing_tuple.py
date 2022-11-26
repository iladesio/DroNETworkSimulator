import numpy as np

from src.routing_algorithms.BASE_routing import BASE_routing
from src.utilities import utilities as util

class QLearningRoutingTuple(BASE_routing):

    def __init__(self, drone, simulator):
        BASE_routing.__init__(self, drone=drone, simulator=simulator)
        self.taken_actions = {}  # id event : [(old_state, old_action), (curr_state, curr_action)]

        self.q_table = {}  # state : list of drones

        # Learning Rate
        self.alpha = 0.6
        # Discount Factor
        self.gamma = 0.8
        # epsilon
        self.epsilon = 0.9

        self._rng = np.random.default_rng()

        # state -> (is_my_next_target_in_depot_range, is_some_packet_expiring_before_nt)
        self.states = [
            (False, False),
            (False, True),
            (True, False),
            (True, True),
        ]

        # init q_table - actions are all drones in the simulator
        for state in self.states:
            self.q_table[state] = [0 for i in range(self.simulator.n_drones)]

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

            # state -> (is_my_next_target_in_depot_range, is_some_packet_expiring_before_nt)
            prev_state, prev_action = self.taken_actions[id_event][0]
            curr_state, curr_action = self.taken_actions[id_event][1]

            print("\nid_event", id_event, "\nTaken action: ", self.taken_actions[id_event])

            """
                step1: taken_action = [id_event: [(state1, action1), (state1, action1)] ] 
                step2: taken_action = [id_event: [(state1, action1), (state2, action2)] ] 
                step3: taken_action = [id_event: [(state2, action2), (state3, action3)] ] 
            """
            # remove the entry, the action has received the feedback
            del self.taken_actions[id_event]

            # reward or update using the old state and the selected action at that time
            # do something or train the model (?)

            # ToDo: Reward function
            if outcome == 1:
                reward = 2
            else:
                reward = -2

            # receive the reward for moving to the new state, and calculate the temporal difference
            q_old_value = self.q_table[prev_state][prev_action]
            temporal_difference = reward + self.gamma * (max(self.q_table[curr_state])) - q_old_value

            print("\nQTable:")
            print(self.q_table)

            # update the Q-value for the previous state and action pair
            new_q_value = q_old_value + (self.alpha * temporal_difference)
            self.q_table[prev_state][prev_action] = new_q_value

    # todo move function @paolopio
    def is_drone_going_to_depot(self, drone):
        # todo implement
        return False

    def relay_selection(self, opt_neighbors: list, packet):
        """
        This function returns the best relay to send packets.
        @param packet:
        @param opt_neighbors: a list of tuple (hello_packet, source_drone)
        @return: The best drone to use as relay
        """

        # Only if you need!
        """
        cell_index = util.TraversedCells.coord_to_cell(size_cell=self.simulator.prob_size_cell,
                                                       width_area=self.simulator.env_width,
                                                       x_pos=self.drone.coords[0],  # e.g. 1500
                                                       y_pos=self.drone.coords[1])[0]  # e.g. 500
        """

        """
            fixed cases:
            - no neighbours (CAPIREEEEEEEEEEEEEEEEE SIUUUUUUMM)
            - 
        """

        # calculate state
        state = (self.is_drone_going_to_depot(self.drone), self.drone.are_packets_expiring_critical())

        neighbors = [n[1] for n in opt_neighbors]

        action = None

        # Strategy: Epsilon-Greedy
        # Decide if the agent should explore or exploit using epsilon
        p = np.random.random()
        if p > self.epsilon:
            should_explore = True
        else:
            should_explore = False

        # Exploit - choose the best action
        if not should_explore:

            # current state - KEEP
            max_score = self.q_table[state][self.drone.identifier]

            for n in neighbors:
                if self.q_table[state][n.identifier] > max_score:
                    action = n
                    max_score = self.q_table[state][n.identifier]

        else:
            # Explore
            # choose a random action
            if len(neighbors) == 0:
                action = None
            else:
                # action = geo.GeoRouting.relay_selection(self, opt_neighbors, packet)
                action = self.simulator.rnd_routing.choice(neighbors)

        if action is None:
            action_identifier = self.drone.identifier
        else:
            action_identifier = action.identifier

        # Store your current action --- you can add some stuff if needed to take a reward later
        if packet.event_ref.identifier not in self.taken_actions:
            previous_action = (state, action_identifier)
        else:
            previous_action = self.taken_actions[packet.event_ref.identifier][1]

        self.taken_actions[packet.event_ref.identifier] = [previous_action, (state, action_identifier)]

        return action