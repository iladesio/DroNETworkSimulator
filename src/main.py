from datetime import datetime

from src.simulation.simulator import Simulator
from src.utilities import config
from src.utilities.config import RoutingAlgorithm


def main():
    if config.SAVE_RUN_METRICS_TXT:

        start_time = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")

        output_filename = config.STATISTICS_RUN_PATH + "/statistics" + start_time + ".txt"

        with open(output_filename, "w") as f:
            f.write("Num drones: " + str(config.N_DRONES))
            f.write("\nSim duration: " + str(config.SIM_DURATION))

            f.write("\nStart: " + datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

            for algo in [RoutingAlgorithm.RND, RoutingAlgorithm.GEO,
                         RoutingAlgorithm.DIR_QL, RoutingAlgorithm.ARDEEP_QL]:
                f.write("\n\nStart run " + algo.name + ": " + datetime.now().strftime("%H:%M:%S"))

                """ the place where to run simulations and experiments. """
                sim = Simulator(
                    routing_algorithm=algo
                )
                sim.run()  # run the simulation
                sim.close()

                metrics = sim.get_metrics()
                for key in metrics.keys():
                    f.write("\n" + key + ": " + str(metrics[key]))

                f.write("\nEnd run " + algo.name + ": " + datetime.now().strftime("%H:%M:%S"))

            f.write("\nEnd: " + datetime.now().strftime("%H:%M:%S"))
    else:
        for algo in [RoutingAlgorithm.ARDEEP_QL]:
            # RoutingAlgorithm.GEO, RoutingAlgorithm.ARDEEP_QL]:
            sim = Simulator(routing_algorithm=algo)
            sim.run()  # run the simulation
            sim.close()


if __name__ == "__main__":
    main()
