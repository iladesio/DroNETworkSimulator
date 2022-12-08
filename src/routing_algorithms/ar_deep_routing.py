import random

from src.routing_algorithms.BASE_routing import BASE_routing
from src.utilities import utilities as util


class ARDeepLearningRouting(BASE_routing):

    # C ui, bj = (ct ui,bj,         expected connection time of the link
    #               PER ui, bj,     Packet Error Ratio of the link
    #               e bj            remaining energy of neighbor bj
    #               d bj, des,      distance between neighbor bj and destination des
    #               d min)          minimum distance between a two hop neighbor bk and des

    def __init__(self, drone, simulator):
        BASE_routing.__init__(self, drone=drone, simulator=simulator)
        self.taken_actions = {}

        self.dq_table = {}

        self.omega = 0  # 0 < w < 1 used to adjust the importance of reliable distance Dij in reward function

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

                current_cell = util.TraversedCells.coord_to_cell(size_cell=self.simulator.prob_size_cell,
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

        """
            hello_packet should contain also the residual energy
        """

        # expected connection time of the link
        connection_time = 0

        # Packet Error Ratio of the link
        packet_error_ratio = 0

        # remaining energy of neighbor bj
        remaining_energy = 0

        # distance between neighbor bj and destination des
        dist_bj_destination = 0

        # minimum distance between a two hop neighbor bk and des
        min_distance_bk_des = 0

        # Strategy: Epsilon-Greedy
        # Decide if the agent should explore or exploit using epsilon

        return None
