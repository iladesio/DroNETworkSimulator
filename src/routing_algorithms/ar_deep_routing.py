import math
import random

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

from src.routing_algorithms.BASE_routing import BASE_routing
from src.routing_algorithms.deep_ql.dqn import DQN
from src.routing_algorithms.deep_ql.replay_memory import ReplayMemory
from src.utilities import utilities as util

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)  # todo remove me

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the AdamW optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4


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

        self.omega = 0  # 0 < w < 1 used to adjust the importance of reliable distance Dij in reward function

        # Constants used in normalization of states
        self.connection_time_max = 0

        self.R_max = 2

        self.connection_time_min = 0  # è costante?

        # dictiory to store (cur_state, taken_action, next_state) for each packet
        # key: event identifier
        # next_state is set to None the first time, and it'll be set after a delta time
        # in the main simulator cycle
        self.taken_actions = {}

        # example parameters
        self.episode_durations = []

        self.n_actions = self.simulator.n_drones - 1  # todo da vedere
        self.n_observations = self.simulator.n_drones - 1  # todo da vedere

        self.policy_net = DQN(self.n_observations, self.n_actions).to(device)
        self.target_net = DQN(self.n_observations, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.memory = ReplayMemory(10000)

    def select_action(self, state, opt_neighbors):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.simulator.cur_step / EPS_DECAY)

        if sample > eps_threshold:
            # exploit
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            # explore
            return self.simulator.rnd_routing.choice([v[1] for v in opt_neighbors])

    def plot_durations(self, show_result=False):
        plt.figure(1)
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())

    # todo colab
    # def optimize_model(self):
    #     if len(memory) < BATCH_SIZE:
    #         return
    #     transitions = memory.sample(BATCH_SIZE)
    #     # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    #     # detailed explanation). This converts batch-array of Transitions
    #     # to Transition of batch-arrays.
    #     batch = Transition(*zip(*transitions))
    #
    #     # Compute a mask of non-final states and concatenate the batch elements
    #     # (a final state would've been the one after which simulation ended)
    #     non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
    #                                             batch.next_state)), device=device, dtype=torch.bool)
    #     non_final_next_states = torch.cat([s for s in batch.next_state
    #                                        if s is not None])
    #     state_batch = torch.cat(batch.state)
    #     action_batch = torch.cat(batch.action)
    #     reward_batch = torch.cat(batch.reward)
    #
    #     # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    #     # columns of actions taken. These are the actions which would've been taken
    #     # for each batch state according to policy_net
    #     state_action_values = policy_net(state_batch).gather(1, action_batch)
    #
    #     # Compute V(s_{t+1}) for all next states.
    #     # Expected values of actions for non_final_next_states are computed based
    #     # on the "older" target_net; selecting their best reward with max(1)[0].
    #     # This is merged based on the mask, such that we'll have either the expected
    #     # state value or 0 in case the state was final.
    #     next_state_values = torch.zeros(BATCH_SIZE, device=device)
    #     with torch.no_grad():
    #         next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    #     # Compute the expected Q values
    #     expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    #
    #     # Compute Huber loss
    #     criterion = nn.SmoothL1Loss()
    #     loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    #
    #     # Optimize the model
    #     optimizer.zero_grad()
    #     loss.backward()
    #     # In-place gradient clipping
    #     torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    #     optimizer.step()

    def get_current_state(self, list_drones):
        state = {}

        for neighbor in list_drones:
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
            dist_bj_destination = np.min(dist_bj_destination / dist_ui_destination, 1)
            min_distance_bk_des = np.min(min_distance_bk_des / dist_ui_destination, 1)

            # state[neighbor.identifier] = (
            state[neighbor.identifier] = (
                connection_time,
                packet_error_ratio,
                remaining_energy,
                dist_bj_destination,
                min_distance_bk_des
            )

        return state

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

            state, action, next_state = self.taken_actions[id_event]

            # todo rivedere il calcolo del local minimum
            local_minimum = True
            for neighbor_state in state:
                # state[3] is dist_bj_destination
                if neighbor_state is not 0 and neighbor_state[3] < 1:
                    local_minimum = False

            state = state[action]

            """
            REWARD FUNCTION
             Rmax, when neighbor bj is the destination
            −Rmax, when neighbor bj is the local minimum
             ω * Dui,bj + (1 − ω) * ( ebj / Ebj ), otherwise
            """
            # when neighbor bj is the destination
            if outcome == 1:  # todo check
                reward = self.R_max
            # when neighbor bj is the local minimum (all neighbors of node ui are further away from the destination than node ui)
            elif local_minimum is True:
                reward = - self.R_max
            else:
                dist_ui_destination = util.euclidean_distance(self.drone.coords, self.drone.depot.coords)
                dist_bj_destination = util.euclidean_distance(action.coord, self.drone.depot.coords)
                connection_time = state[0]
                packet_error_ratio = state[1]
                remaining_energy = state[2]

                if connection_time >= self.connection_time_min:
                    beta = 1
                else:
                    beta = 0

                D_ui_bj = dist_ui_destination / dist_bj_destination * (1 - packet_error_ratio) * beta

                reward = self.omega * D_ui_bj + (1 - self.omega) * remaining_energy

            # save sample in experience replay memory
            self.memory.push(state, action, next_state, reward)

            # remove the entry, the action has received the feedback
            del self.taken_actions[id_event]

    def relay_selection(self, opt_neighbors: list, packet):

        """
        This function returns the best relay to send packets.
        @param packet:
        @param opt_neighbors: a list of tuple (hello_packet, source_drone)
        @return: The best drone to use as relay
        """

        # todo: hello_packet should contain also the residual energy

        list_neighbors = [n[1] for n in opt_neighbors]
        state = self.get_current_state(list_neighbors)

        chosen_action = self.select_action(state, opt_neighbors)

        # build the tuple with each state starting from the complete list of drones
        # i.e. [0, (...state...), 0, 0, (...state...) ]
        complete_state = [state[drone.identifier] if drone in list_neighbors else 0 for drone in self.simulator.drones]

        # store in taken actions
        self.taken_actions[packet.event_ref.identifier] = (complete_state, chosen_action.identifier, None)

        return chosen_action
