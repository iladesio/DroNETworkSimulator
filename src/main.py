
from timeit import default_timer as timer

from src.simulation.simulator import Simulator


def main():
    """ the place where to run simulations and experiments. """
    start = timer()

    sim = Simulator()  # empty constructor means that all the parameters of the simulation are taken from src.utilities.config.py
    sim.run()  # run the simulation
    sim.close()

    print("Start: ", start)
    print("End: ", timer() - start)

if __name__ == "__main__":
    main()
