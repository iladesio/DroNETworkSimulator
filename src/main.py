from datetime import datetime

from src.simulation.simulator import Simulator
from src.utilities import config


def main():
    if config.SAVE_ARDEEP_METRICS_TXT:

        start_time = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")

        output_filename = config.STATISTICS_RUN_PATH + "/statistics" + start_time + ".txt"

        with open(output_filename, "w") as f:
            f.write("Num drones: " + str(config.N_DRONES))
            f.write("\nSim duration: " + str(config.SIM_DURATION))

            f.write("\nStart: " + datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

            for i in range(1):
                f.write("\n\nStart run " + str(i) + ": " + datetime.now().strftime("%H:%M:%S"))

                """ the place where to run simulations and experiments. """
                sim = Simulator()  # empty constructor means that all the parameters of the simulation are taken from src.utilities.config.py
                sim.run()  # run the simulation
                sim.close()

                metrics = sim.get_metrics()
                for key in metrics.keys():
                    f.write("\n" + key + ": " + str(metrics[key]))

                f.write("\nEnd run " + str(i) + ": " + datetime.now().strftime("%H:%M:%S"))

            f.write("\nEnd: " + datetime.now().strftime("%H:%M:%S"))
    else:
        """ the place where to run simulations and experiments. """
        sim = Simulator()  # empty constructor means that all the parameters of the simulation are taken from src.utilities.config.py
        sim.run()  # run the simulation
        sim.close()


if __name__ == "__main__":
    main()
