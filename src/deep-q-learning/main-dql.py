import random

import gym
import numpy as np
import progressbar
from IPython.display import clear_output
from tensorflow.keras.optimizers import Adam

from agent import Agent


def main():
    """ Simulazione """
    enviroment = gym.make("Taxi-v3")

    print('Number of states: {}'.format(enviroment.observation_space.n))
    print('Number of actions: {}'.format(enviroment.action_space.n))

    """ 
        Adam optimization is a stochastic gradient descent method that is based on
        adaptive estimation of first-order and second-order moments.
    """
    optimizer = Adam(learning_rate=0.01)

    """ Drone """
    agent = Agent(enviroment, optimizer)

    alpha = 0.1
    gamma = 0.6
    epsilon = 0.1
    q_table = np.zeros([enviroment.observation_space.n, enviroment.action_space.n])

    batch_size = 32
    num_of_episodes = 3
    timesteps_per_episode = 3

    for episode in range(0, num_of_episodes):
        # Reset the enviroment
        state = enviroment.reset()
        state = state[0]

        # Initialize variables
        reward = 0
        terminated = False

        while not terminated:
            # Take learned path or explore new actions based on the epsilon
            if random.uniform(0, 1) < epsilon:
                action = enviroment.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            # Take action
            next_state, reward, terminated, truncated, info = enviroment.step(action)

            # Recalculate
            q_value = q_table[state, action]
            max_value = np.max(q_table[next_state])
            new_q_value = (1 - alpha) * q_value + alpha * (reward + gamma * max_value)

            # Update Q-table
            q_table[state, action] = new_q_value
            state = next_state

        if (episode + 1) % 100 == 0:
            clear_output(wait=True)
            print("Episode: {}".format(episode + 1))
            # enviroment.render()

    print("**********************************")
    print("Training is done!\n")
    print("**********************************")

    total_epochs = 0
    total_penalties = 0
    num_of_episodes = 100

    for _ in range(num_of_episodes):
        state = enviroment.reset()[0]
        epochs = 0
        penalties = 0
        reward = 0

        terminated = False
        truncated = False

        while not terminated and not truncated:
            action = np.argmax(q_table[state])
            state, reward, terminated, truncated, info = enviroment.step(action)

            if reward == -10:
                penalties += 1

            epochs += 1

        total_penalties += penalties
        total_epochs += epochs

    print("**********************************")
    print("Results")
    print("**********************************")
    print("Epochs per episode: {}".format(total_epochs / num_of_episodes))
    print("Penalties per episode: {}".format(total_penalties / num_of_episodes))

    agent.q_network.summary()

    for e in range(0, num_of_episodes):
        # Reset the enviroment
        state = enviroment.reset()[0]
        state = np.reshape(state, [1, 1])

        # Initialize variables
        reward = 0
        terminated = False

        bar = progressbar.ProgressBar(maxval=timesteps_per_episode,
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()

        for timestep in range(timesteps_per_episode):
            # Run Action
            action = agent.act(state)

            # Take action
            next_state, reward, terminated, truncated, info = enviroment.step(action)
            next_state = np.reshape(next_state, [1, 1])
            agent.store(state, action, reward, next_state, terminated)

            state = next_state

            if terminated or truncated:
                agent.alighn_target_model()
                break

            if len(agent.expirience_replay) > batch_size:
                agent.retrain(batch_size)

            bar.update(timestep)

        bar.finish()

        print("**********************************")
        print("Episode: {}".format(e + 1))
        print("**********************************")

    agent.q_network.summary()


if __name__ == "__main__":
    main()
