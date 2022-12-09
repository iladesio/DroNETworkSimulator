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

        # use this to have always the same simulation
        random.seed(self.simulator.seed)

        self.taken_actions = {}

        self.dq_table = {}

        self.omega = 0  # 0 < w < 1 used to adjust the importance of reliable distance Dij in reward function

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

            """
            REWARD FUNCTION
            Rmax, when neighbor bj is the destination
            −Rmax, when neighbor bj is the local minimum
            ω * Dui,bj + (1 − ω) * ( ebj / Ebj ), otherwise
            """

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

        states = {}

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

            states[neighbor.identifier] = (
                connection_time,
                packet_error_ratio,
                remaining_energy,
                dist_bj_destination,
                min_distance_bk_des
            )

        chosen_state = states[0]
        chosen_action = None

        # todo calculate next state and choose next drone
        self.taken_actions[packet.event_ref.identifier] = (chosen_state, chosen_action)

        return None
