"""
Now, let's define our model. But first, let's quickly recap what a DQN is.

DQN algorithm
-------------

Our environment is deterministic, so all equations presented here are
also formulated deterministically for the sake of simplicity. In the
reinforcement learning literature, they would also contain expectations
over stochastic transitions in the environment.

Our aim will be to train a policy that tries to maximize the discounted,
cumulative reward
:math:`R_{t_0} = \sum_{t=t_0}^{\infty} \gamma^{t - t_0} r_t`, where
:math:`R_{t_0}` is also known as the *return*. The discount,
:math:`\gamma`, should be a constant between :math:`0` and :math:`1`
that ensures the sum converges. A lower :math:`\gamma` makes
rewards from the uncertain far future less important for our agent
than the ones in the near future that it can be fairly confident
about. It also encourages agents to collect reward closer in time
than equivalent rewards temporally future away.

The main idea behind Q-learning is that if we had a function
:math:`Q^*: State \times Action \rightarrow \mathbb{R}`, that could tell
us what our return would be, if we were to take an action in a given
state, then we could easily construct a policy that maximizes our
rewards:

.. math:: \pi^*(s) = \arg\!\max_a \ Q^*(s, a)

However, we don't know everything about the world, so we don't have
access to :math:`Q^*`. But, since neural networks are universal function
approximators, we can simply create one and train it to resemble
:math:`Q^*`.

For our training update rule, we'll use a fact that every :math:`Q`
function for some policy obeys the Bellman equation:

.. math:: Q^{\pi}(s, a) = r + \gamma Q^{\pi}(s', \pi(s'))

The difference between the two sides of the equality is known as the
temporal difference error, :math:`\delta`:

.. math:: \delta = Q(s, a) - (r + \gamma \max_a' Q(s', a))

To minimise this error, we will use the `Huber
loss <https://en.wikipedia.org/wiki/Huber_loss>`__. The Huber loss acts
like the mean squared error when the error is small, but like the mean
absolute error when the error is large - this makes it more robust to
outliers when the estimates of :math:`Q` are very noisy. We calculate
this over a batch of transitions, :math:`B`, sampled from the replay
memory:

.. math::

   \mathcal{L} = \frac{1}{|B|}\sum_{(s, a, s', r) \ \in \ B} \mathcal{L}(\delta)

.. math::

   \text{where} \quad \mathcal{L}(\delta) = \begin{cases}
     \frac{1}{2}{\delta^2}  & \text{for } |\delta| \le 1, \\
     |\delta| - \frac{1}{2} & \text{otherwise.}
   \end{cases}

Q-network
^^^^^^^^^

Our model will be a convolutional neural network that takes in the
difference between the current and previous screen patches. It has two
outputs, representing :math:`Q(s, \mathrm{left})` and
:math:`Q(s, \mathrm{right})` (where :math:`s` is the input to the
network). In effect, the network is trying to predict the *expected return* of
taking each action given the current input.

"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utilities import config


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch during optimization.
    # Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        mask = self.get_mask(x).to(config.DEVICE)
        x = F.leaky_relu(self.layer1(x), negative_slope=0.4)
        x = F.leaky_relu(self.layer2(x), negative_slope=0.4)
        x = self.layer3(x)
        result = F.softmax(torch.masked_fill(x, mask == 0, float('-inf')), dim=1)
        return result

    def get_mask(self, state):
        complete_mask = []

        for row in state:
            mask = []

            splitted_tensors = torch.split(row, 5)

            for statino in splitted_tensors:
                if torch.sum(statino).item() == 0.:
                    mask.append(0)
                else:
                    mask.append(1)

            complete_mask.append(mask)

        return torch.Tensor(complete_mask)
