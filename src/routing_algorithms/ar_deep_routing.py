import random
import numpy as np

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

        # use this to have always the same simulation
        random.seed(self.simulator.seed)

        self.taken_actions = {}

        self.dq_table = {}

        self.omega = 0  # 0 < w < 1 used to adjust the importance of reliable distance Dij in reward function

        # Constants used in normalization of states:
        self.connection_time_max = 0

        self.R_max = 2

        connection_time_min = 0 # è costante?

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

            state, action = self.taken_actions[id_event]

            local_minimum = True
            for n in state:
                if n[3] < 1:
                    local_minimum = False

            state = state[action]

            """
            REWARD FUNCTION
            Rmax, when neighbor bj is the destination
            −Rmax, when neighbor bj is the local minimum
            ω * Dui,bj + (1 − ω) * ( ebj / Ebj ), otherwise
            """
            # when neighbor bj is the destination
            if outcome == 1:    #non sono sicura
                reward = self.R_max
            # when neighbor bj is the local minimum (all neighbors of node ui are further away from the destination than node ui)
            elif local_minimum is True:
                reward = - self.R_max
            else:
                dist_ui_destination = util.euclidean_distance(self.drone.coords, self.drone.depot.coords)
                dist_bj_destination = util.euclidean_distance(action.coord, self.self.drone.depot.coords)
                connection_time = state[0]
                packet_error_ratio = state[1]
                remaining_energy = state[2]

                if connection_time >= self.connection_time_min:
                    beta = 1
                else:
                    beta = 0

                D_ui_bj = dist_ui_destination / dist_bj_destination * (1 - packet_error_ratio) * beta

                reward = self.omega * D_ui_bj + (1 - self.omega) * remaining_energy


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

        list_neighbors = [n[1] for n in opt_neighbors]

        state = {}

        for neighbor in list_neighbors:
            # expected connection time of the link
            # todo capire come calcolare
            connection_time = 100

            # Packet Error Ratio of the link - generated randomly between 0 and 0.2
            packet_error_ratio = random.uniform(0, 0.2)

            # remaining energy of neighbor bj
            remaining_energy = neighbor.residual_energy

            # distance between neighbor bj and destination des
            dist_bj_destination = util.euclidean_distance(neighbor.coord, self.simulator.depot_coordinates)

            # minimum distance between a two hop neighbor bk and des
            # todo capire come calcolare
            min_distance_bk_des = 100

            # C ui, bj = (ct ui,bj,         expected connection time of the link
            #               PER ui, bj,     Packet Error Ratio of the link
            #               e bj            remaining energy of neighbor bj
            #               d bj, des,      distance between neighbor bj and destination des
            #               d min)          minimum distance between a two hop neighbor bk and des

            # Normalization in range [0, 1] of all the elements of the state
            connection_time = connection_time / self.connection_time_max
            remaining_energy = remaining_energy / self.simulator.drone_max_energy
            dist_ui_destination = util.euclidean_distance(self.drone.coords, self.drone.depot.coords)
            dist_bj_destination = np.min(dist_bj_destination/dist_ui_destination, 1)
            min_distance_bk_des = np.min(min_distance_bk_des/dist_ui_destination, 1)

            # state[neighbor.identifier] = (
            state[neighbor] = (
                connection_time,
                packet_error_ratio,
                remaining_energy,
                dist_bj_destination,
                min_distance_bk_des
            )

        chosen_state = state[0]
        chosen_action = None

        # todo calculate next state and choose next drone
        self.taken_actions[packet.event_ref.identifier] = (chosen_state, chosen_action)

        return None
