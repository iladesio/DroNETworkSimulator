from src.routing_algorithms.BASE_routing import BASE_routing
import src.utilities.utilities as util

class GeoRouting(BASE_routing):

    def __init__(self, drone, simulator):
        BASE_routing.__init__(self, drone, simulator)

    def relay_selection(self, opt_neighbors, packet):
        """
        This function returns a relay for packets according to geographic routing using C2S criteria.

        @param packet:
        @param opt_neighbors: a list of tuples (hello_packet, drone)
        @return: The best drone to use as relay or None if no relay is selected
        """

        d0_coords = self.drone.next_target()

        drone_to_send = None

        distance_d0_to_depot = util.euclidean_distance(d0_coords, self.drone.depot.coords)
        best_distance = distance_d0_to_depot

        for hpk, d_drone in opt_neighbors:
            d_pos = hpk.next_target

            distance_d_to_depot = util.euclidean_distance(d_pos, self.drone.depot.coords)

            if distance_d_to_depot < best_distance:
                drone_to_send = d_drone
                best_distance = distance_d_to_depot

        return drone_to_send
