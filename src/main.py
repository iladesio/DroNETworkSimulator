import json

from src.experiments.parser.parser import command_line_parser
from src.simulation.simulator import Simulator
from src.utilities import config, utilities
from src.utilities.config import RoutingAlgorithm


def main():
    """ the place where to run simulations and experiments. """

    args = command_line_parser.parse_args()

    initial_seed = args.initial_seed
    end_seed = args.end_seed
    num_drones = args.number_of_drones

    map_alg = {
        'RND': RoutingAlgorithm.RND,
        'GEO': RoutingAlgorithm.GEO,
        'DIR_QL': RoutingAlgorithm.DIR_QL,
    }

    if initial_seed is None and end_seed is None and num_drones is None:
        print("base execution")
        sim = Simulator()  # empty constructor means that all the parameters of the simulation are taken from src.utilities.config.py
        sim.run()  # run the simulation
        sim.close()

    else:

        if initial_seed is not None and end_seed is not None:

            simulation_results = {}
            n_drones = num_drones if num_drones is not None else config.N_DRONES

            for seed in range(initial_seed, end_seed + 1):
                print("\nrunning seed: ", seed)

                sim = Simulator(len_simulation=config.SIM_DURATION,
                                time_step_duration=config.TS_DURATION,
                                seed=seed,  # current seed
                                n_drones=n_drones,
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
                                simulation_name="")
                sim.run()  # run the simulation

                simulation_results[seed] = sim.metrics.get_metrics()

                sim.close()

            simulation_name = "experiment_simulation-" + utilities.date() + "_"
            filename = (config.ROOT_EVALUATION_DATA + simulation_name + ".json")

            js = json.dumps(simulation_results)
            f = open(filename, "w")
            f.write(js)
            f.close()

            for seed in simulation_results.keys():
                print(seed, " : ", simulation_results[seed]["packet_delivery_ratio"])

        else:
            """
            simulation_results = {}

            for alg in ["RND", "GEO", "DIR_QL"]:
                print("ALGORITM: ", alg)
                algoritm_results = {}

                for num_drone in [5, 10, 15, 20, 25, 40, 50]:
                    print("NUM DRONE: ", num_drone)

                    sim = Simulator(len_simulation=config.SIM_DURATION,
                                    time_step_duration=config.TS_DURATION,
                                    seed=config.SEED,
                                    n_drones=num_drone,
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
                                    routing_algorithm=map_alg[alg],
                                    communication_error_type=config.CHANNEL_ERROR_TYPE,
                                    prob_size_cell_r=config.CELL_PROB_SIZE_R,
                                    simulation_name="")
                    sim.run()  # run the simulation
                    algoritm_results[num_drone] = sim.metrics.get_metrics()

                    sim.close()

                simulation_results[alg] = algoritm_results

            simulation_name = "simulation_metrics" + utilities.date() + "_"
            filename = (config.ROOT_EVALUATION_DATA + simulation_name + ".json")

            js = json.dumps(simulation_results)
            f = open(filename, "w")
            f.write(js)
            f.close()

            print(simulation_results)

            """
            simulation_results = {}

            for num_steps in [10, 15, 20, 30, 40, 50, 70, 80, 90, 100, 150, 200, 300]:
                print("LEN_ SIMULATION: ", num_steps)

                sim = Simulator(len_simulation=num_steps * 1000,
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
                                simulation_name="")
                sim.run()  # run the simulation
                sim.close()

                simulation_results[num_steps] = sim.metrics.get_metrics()

            simulation_name = "num_steps_simulation_metrics" + utilities.date() + "_"
            filename = (config.ROOT_EVALUATION_DATA + simulation_name + ".json")

            js = json.dumps(simulation_results)
            f = open(filename, "w")
            f.write(js)
            f.close()

            print(simulation_results)


if __name__ == "__main__":
    main()
