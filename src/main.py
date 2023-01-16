from datetime import datetime

import wandb

from src.simulation.simulator import Simulator
from src.utilities import config
from src.utilities.config import RoutingAlgorithm


def main():
    if config.SAVE_RUN_METRICS_TXT:

        start_time = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")

        output_filename = config.STATISTICS_RUN_PATH + "/statistics" + start_time + ".txt"

        for sim_duration in [50]:
            for n_drones in [10, 15, 20, 25, 30]:

                with open(output_filename, "w") as f:
                    f.write("Num drones: " + str(n_drones))
                    f.write("\nSim duration: " + str(sim_duration))

                    f.write("\nStart: " + datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

                    for algo in [RoutingAlgorithm.ARDEEP_QL]:

                        f.write("\n\nStart run " + algo.name + ": " + datetime.now().strftime("%H:%M:%S"))

                        """ the place where to run simulations and experiments. """
                        sim = Simulator(
                            routing_algorithm=algo,
                            n_drones=n_drones,
                            len_simulation=sim_duration * 1000
                        )
                        sim.run()  # run the simulation
                        sim.close()

                        metrics = sim.get_metrics()
                        for key in metrics.keys():
                            f.write("\n" + key + ": " + str(metrics[key]))

                        f.write("\nEnd run " + algo.name + ": " + datetime.now().strftime("%H:%M:%S"))

                    f.write("\nEnd: " + datetime.now().strftime("%H:%M:%S"))

    elif config.RUN_WITH_SWEEP:

        learning_rate_min = config.LR / (1 + config.SIM_DURATION / config.LR_DEC_SPEED)

        sweep_configuration = {
            'project': 'ardeep_ql',
            'name': 'sweep_pina',
            'method': 'bayes',
            'metric': {
                'goal': 'minimize',
                'name': 'loss'
            },
            'parameters': {
                'batch_size': {'values': [64, 128, 256, 512]},
                'learning_rate': {'max': config.LR, 'min': learning_rate_min}
            }
        }

        sim = Simulator()

        sweep_id = wandb.sweep(sweep_configuration, project="ardeep_ql")

        # start sweep agent
        wandb.agent(sweep_id, function=sim.run, count=10)

        sim.run()  # run the simulation
        sim.close()

    else:
        sim = Simulator(routing_algorithm=RoutingAlgorithm.ARDEEP_QL)
        sim.run()  # run the simulation
        sim.close()


if __name__ == "__main__":
    main()
