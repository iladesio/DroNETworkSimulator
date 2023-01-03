from enum import Enum

import torch

from src.routing_algorithms.ar_deep_routing import ARDeepLearningRouting
from src.routing_algorithms.georouting import GeoRouting
from src.routing_algorithms.q_learning_direction import QLearningRoutingDirection
from src.routing_algorithms.random_routing import RandomRouting

"""
This file contains all the constants and parameters of the simulator.
It comes handy when you want to make one shot simulations, making parameters and constants vary in every
simulation. For an extensive experimental campaign read the header at src.simulator.

Attributes that one needs tweak often are tagged with # ***
"""

# ----------------------------------------------------------------------------------
#
#                  ██████  ██████  ███    ██ ███████ ██  ██████ 
#                 ██      ██    ██ ████   ██ ██      ██ ██      
#                 ██      ██    ██ ██ ██  ██ █████   ██ ██   ███ 
#                 ██      ██    ██ ██  ██ ██ ██      ██ ██    ██ 
#                  ██████  ██████  ██   ████ ██      ██  ██████  
#
# ----------------------------------------------------------------------------------

# ----------------------- PATH DRONES -----------------------------------------#
CIRCLE_PATH = False  # bool: whether to use cirlce paths around the depot
DEMO_PATH = False  # bool: whether to use handcrafted tours or not
# to set up handcrafted torus see utilities.utilities
PATH_FROM_JSON = True  # bool: whether to use the path (for drones) store in the JSONS_PATH_PREFIX,
# otherwise path are generated online
JSONS_PATH_PREFIX = "data/tours/RANDOM_missions0.json"  # str: the path to the drones tours,
# the {} should be used to specify the seed -> es. data/tours/RANDOM_missions1.json for seed 1.
RANDOM_STEPS = [250, 500, 700, 900, 1100,
                1400]  # the step after each new random directions is taken, in case of dynamic generation
RANDOM_START_POINT = True  # bool whether the drones start the mission at random positions

# ------------------------------- CONSTANTS ------------------------------- #

STATISTICS_RUN_PATH = "data/statistics"

""" ---------------- AR_DEEP constants ---------------- """
CALCULATE_NEXT_STATE_DELTA = 50
TRAINING_DELTA = 300

SAVE_ARDEEP_METRICS_TXT = False  # bool: output all simulation metrics and times in a txt file

# current data -> sim_duration: 200_000 n_drones: 20
CONNECTION_TIME_JSON = "data/ar-deep/connection_time.json"  # str: the path to connection time data
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# output file path of the ReplayMemory
REPLAY_MEMORY_JSON = "data/ar-deep/replay_memory.json"
REPLAY_MEMORY_PT = "data/ar-deep/replay_memory.pt"

SAVE_MODEL_DICT = False  # bool: save model dict after training
READ_MODEL_DICT = False  # bool: save model dict after training
# file path of the model (saved after training and read before run the simulation)
TRAIN_MODEL = False  # bool: call optimize_model() fnc to train model on simulator close
MODEL_STATE_DICT_PATH = "data/ar-deep/model_state_dict.pt"

SAVE_CONNECTION_TIME_DATA = True  # bool: save connection time data to the file
USE_CONNECTION_TIME_DATA = True  # bool: read connection time data from the file

DEBUG = False  # bool: whether to print debug strings or not.
EXPERIMENTS_DIR = "data/evaluation_tests/"  # output data : the results of the simulation

# drawaing
PLOT_SIM = False  # bool: whether to plot or not the simulation.
WAIT_SIM_STEP = 0  # .1     # float: seconds, pauses the rendering for 'DELAY_PLOT' seconds.
SKIP_SIM_STEP = 10  # int: steps, plot the simulation every 'RENDERING_STEP' steps. At least 1.
DRAW_SIZE = 600  # int: size of the drawing window.
IS_SHOW_NEXT_TARGET_VEC = True  # bool : whether show the direction and next target of the drone

SAVE_PLOT = False  # bool: whether to save the plots of the simulation or not.
SAVE_PLOT_DIR = "data/plots/"

# add constants here...

# ----------------------------- SIMULATION PARAMS. ---------------------------- #
SIM_DURATION = 20_000  # int: steps of simulation. # ***
TS_DURATION = 0.150  # float: seconds duration of a step in seconds.
SEED = 10  # int: seed of this simulation.

N_DRONES = 10  # int: number of drones. # ***
CONNECTION_TIME_MAX = SIM_DURATION  # default value when we don't have data

ENV_WIDTH = 1500  # float: meters, width of environment.
ENV_HEIGHT = 1500  # float: meters, height of environment.

# events
EVENTS_DURATION = 1500  # SIM_DURATION  # int: steps, number of time steps that an event lasts  -> to seconds = step * step_duration.
D_FEEL_EVENT = 65  # int: steps, a new packet is felt (generated on the drone) every 'D_FEEL_EVENT' steps. # ***
P_FEEL_EVENT = .8  # float: probability that the drones feels the event generated on the drone. # ***

""" e.g. given D_FEEL_EVENT = 500, P_FEEL_EVENT = .5, every 500 steps with probability .5 the drone will feel an event."""

# drones
COMMUNICATION_RANGE_DRONE = 150  # float: meters, communication range of the drones.
SENSING_RANGE_DRONE = 0  # float: meters, the sensing range of the drones.
DRONE_SPEED = 8  # float: m/s, drone speed.
DRONE_MAX_BUFFER_SIZE = 10000  # int: max number of packets in the buffer of a drone.
DRONE_MAX_ENERGY = SIM_DURATION / 10  # 1000000  # int: max energy of a drone.

# depot
DEPOT_COMMUNICATION_RANGE = 150  # float: meters, communication range of the depot.
DEPOT_COO = (750, 0)  # (float, float): coordinates of the depot.


# ------------------------------- ROUTING PARAMS. ------------------------------- #
class RoutingAlgorithm(Enum):
    GEO = GeoRouting
    RND = RandomRouting
    DIR_QL = QLearningRoutingDirection
    ARDEEP_QL = ARDeepLearningRouting

    @staticmethod
    def keylist():
        return list(map(lambda c: c.name, RoutingAlgorithm))


class ChannelError(Enum):
    UNIFORM = 1
    GAUSSIAN = 2
    NO_ERROR = 3

    @staticmethod
    def keylist():
        return list(map(lambda c: c.name, ChannelError))


ROUTING_ALGORITHM = RoutingAlgorithm.ARDEEP_QL

CHANNEL_ERROR_TYPE = ChannelError.GAUSSIAN

COMMUNICATION_P_SUCCESS = 1  # float: probability to have success in a communication.
GUASSIAN_SCALE = .9  # float [0,1]: scale the error probability of the guassian -> success * GUASSIAN_SCALER
PACKETS_MAX_TTL = 200  # float: threshold in the maximum number of hops. Causes loss of packets.
RETRANSMISSION_DELAY = 10  # int: how many time steps to wait before transmit again (for k retransmissions). # ---  #delta_k

# ------------------------------------------- ROUTING MISC --------------------------------- #
HELLO_DELAY = 5  # int : how many time steps wait before transmit again an hello message
RECEPTION_GRANTED = 0.95  # float : the min amount of success to evalute a neigh as relay
LIL_DELTA = 1  # INT:  > 0
OLD_HELLO_PACKET = 50

ROOT_EVALUATION_DATA = "data/evaluation_tests/"

NN_MODEL_PATH = "data/nnmodels/"

# --------------- new cell probabilities -------------- #
CELL_PROB_SIZE_R = 3.75  # the percentage of cell size with respect to drone com range
ENABLE_PROBABILITIES = False

PACKETS_EXPIRING_THRESHOLD = 0.1
