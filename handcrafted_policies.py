import numpy as np

N_CHANNELS = 5
N_ACTIONS = 4
NORTH, EAST, SOUTH, WEST = np.arange(4)
GREAT, GOOD, BAD = np.arange(3)  # Channels


hc_weight_tensor_by_obsho = dict()
hc_bias_tensor_by_obsho = dict()


def make_deterministic_policy(weight_tensor, bias_tensor):
    def policy(observation):
        obs = observation.reshape(1, -1)
        w = weight_tensor.reshape(-1, N_ACTIONS)
        logits = obs.dot(w) + bias_tensor
        best_action = np.argmax(logits)
        return np.eye(N_ACTIONS)[best_action]

    return policy


def uniform_random_policy(observation):
    return np.ones(N_ACTIONS) / N_ACTIONS

# These policies are surely not optimal policies
# Instead, they are what can easily be set by hand
# Known issues:
# - circling / going back and forth
#   - stochasticity could help in this case
#   - having a preferred direction helps (this is the case in general, but does not always work)
#   - "good" objects are not avoided if it is on the path to a better one

# --- observation horizon 1 ---
hc_weight_tensor_by_obsho[1] = np.zeros(
    (3, 3, N_CHANNELS, N_ACTIONS)
)

# DIRECT PICKUPS
hc_weight_tensor_by_obsho[1][0, 1, GREAT, NORTH] = 14 * 10**7
hc_weight_tensor_by_obsho[1][1, 2, GREAT, EAST] = 13 * 10**7
hc_weight_tensor_by_obsho[1][2, 1, GREAT, SOUTH] = 12 * 10**7
hc_weight_tensor_by_obsho[1][1, 0, GREAT, WEST] = 11 * 10**7

hc_weight_tensor_by_obsho[1][0, 1, GOOD, NORTH] = 14 * 10**4
hc_weight_tensor_by_obsho[1][1, 2, GOOD, EAST] = 13 * 10**4
hc_weight_tensor_by_obsho[1][2, 1, GOOD, SOUTH] = 12 * 10**4
hc_weight_tensor_by_obsho[1][1, 0, GOOD, WEST] = 11 * 10**4

hc_weight_tensor_by_obsho[1][0, 1, BAD, NORTH] = -10**9
hc_weight_tensor_by_obsho[1][1, 2, BAD, EAST] = -10**9
hc_weight_tensor_by_obsho[1][2, 1, BAD, SOUTH] = -10**9
hc_weight_tensor_by_obsho[1][1, 0, BAD, WEST] = -10**9


# Diagonals
hc_weight_tensor_by_obsho[1][0, 2, GREAT, NORTH] = 18 * 10 ** 6
hc_weight_tensor_by_obsho[1][0, 2, GREAT, EAST] = 17 * 10 ** 6

hc_weight_tensor_by_obsho[1][2, 2, GREAT, EAST] = 16 * 10 ** 6
hc_weight_tensor_by_obsho[1][2, 2, GREAT, SOUTH] = 15 * 10 ** 6

hc_weight_tensor_by_obsho[1][2, 0, GREAT, SOUTH] = 14 * 10 ** 6
hc_weight_tensor_by_obsho[1][2, 0, GREAT, WEST] = 13 * 10 ** 6

hc_weight_tensor_by_obsho[1][0, 0, GREAT, WEST] = 12 * 10 ** 6
hc_weight_tensor_by_obsho[1][0, 0, GREAT, NORTH] = 11 * 10 ** 6


hc_weight_tensor_by_obsho[1][0, 2, GOOD, NORTH] = 18 * 10 ** 3
hc_weight_tensor_by_obsho[1][0, 2, GOOD, EAST] = 17 * 10 ** 3

hc_weight_tensor_by_obsho[1][2, 2, GOOD, EAST] = 16 * 10 ** 3
hc_weight_tensor_by_obsho[1][2, 2, GOOD, SOUTH] = 15 * 10 ** 3

hc_weight_tensor_by_obsho[1][2, 0, GOOD, SOUTH] = 14 * 10 ** 3
hc_weight_tensor_by_obsho[1][2, 0, GOOD, WEST] = 13 * 10 ** 3

hc_weight_tensor_by_obsho[1][0, 0, GOOD, WEST] = 12 * 10 ** 3
hc_weight_tensor_by_obsho[1][0, 0, GOOD, NORTH] = 11 * 10 ** 3


# Go circular at walls
hc_weight_tensor_by_obsho[1][0, 1, BAD, EAST] = 8
hc_weight_tensor_by_obsho[1][1, 2, BAD, SOUTH] = 8
hc_weight_tensor_by_obsho[1][2, 1, BAD, WEST] = 8
hc_weight_tensor_by_obsho[1][1, 0, BAD, NORTH] = 8
# Diagonal
hc_weight_tensor_by_obsho[1][0, 2, BAD, EAST] = 4
hc_weight_tensor_by_obsho[1][2, 2, BAD, WEST] = 4
hc_weight_tensor_by_obsho[1][2, 0, BAD, WEST] = 4
hc_weight_tensor_by_obsho[1][0, 0, BAD, EAST] = 4


hc_bias_tensor_by_obsho[1] = np.array([1, 0, -1, -2])


# --- observation horizon 2 ---
hc_weight_tensor_by_obsho[2] = np.zeros(
    (5, 5, N_CHANNELS, N_ACTIONS)
)
# First, copy the one for obsho 1
hc_weight_tensor_by_obsho[2][1:-1, 1:-1, :, :] = hc_weight_tensor_by_obsho[1]

# Two-hops to big one: less than one-hop to big one, less than avoiding bad one, more than one-hop small one
# 01234
hc_weight_tensor_by_obsho[2][0, 2, GREAT, NORTH] = 10 ** 6
hc_weight_tensor_by_obsho[2][2, 4, GREAT, EAST] = 10 ** 6
hc_weight_tensor_by_obsho[2][4, 2, GREAT, SOUTH] = 10 ** 6
hc_weight_tensor_by_obsho[2][2, 0, GREAT, WEST] = 10 ** 6

hc_weight_tensor_by_obsho[2][0, 2, GOOD, NORTH] = 10 ** 2
hc_weight_tensor_by_obsho[2][2, 4, GOOD, EAST] = 10 ** 2
hc_weight_tensor_by_obsho[2][4, 2, GOOD, SOUTH] = 10 ** 2
hc_weight_tensor_by_obsho[2][2, 0, GOOD, WEST] = 10 ** 2


# Diagonals
hc_weight_tensor_by_obsho[2][0, 4, GREAT, NORTH] = 10 ** 6
hc_weight_tensor_by_obsho[2][0, 4, GREAT, EAST] = 10 ** 6

hc_weight_tensor_by_obsho[2][4, 4, GREAT, EAST] = 10 ** 6
hc_weight_tensor_by_obsho[2][4, 4, GREAT, SOUTH] = 10 ** 6

hc_weight_tensor_by_obsho[2][4, 0, GREAT, SOUTH] = 10 ** 6
hc_weight_tensor_by_obsho[2][4, 0, GREAT, WEST] = 10 ** 6

hc_weight_tensor_by_obsho[2][0, 0, GREAT, WEST] = 10 ** 6
hc_weight_tensor_by_obsho[2][0, 0, GREAT, NORTH] = 10 ** 6


hc_weight_tensor_by_obsho[2][0, 4, GOOD, NORTH] = 10 ** 3
hc_weight_tensor_by_obsho[2][0, 4, GOOD, EAST] = 10 ** 3

hc_weight_tensor_by_obsho[2][4, 4, GOOD, EAST] = 10 ** 3
hc_weight_tensor_by_obsho[2][4, 4, GOOD, SOUTH] = 10 ** 3

hc_weight_tensor_by_obsho[2][4, 0, GOOD, SOUTH] = 10 ** 3
hc_weight_tensor_by_obsho[2][4, 0, GOOD, WEST] = 10 ** 3

hc_weight_tensor_by_obsho[2][0, 0, GOOD, WEST] = 10 ** 3
hc_weight_tensor_by_obsho[2][0, 0, GOOD, NORTH] = 10 ** 3


# HALF DIAGONALS
hc_weight_tensor_by_obsho[2][0, 1, GREAT, NORTH] = 5 * 10 ** 6
hc_weight_tensor_by_obsho[2][0, 1, GREAT, WEST] = 4 * 10 ** 6

hc_weight_tensor_by_obsho[2][0, 3, GREAT, NORTH] = 5 * 10 ** 6
hc_weight_tensor_by_obsho[2][0, 3, GREAT, EAST] = 4 * 10 ** 6

hc_weight_tensor_by_obsho[2][1, 4, GREAT, EAST] = 5 * 10 ** 6
hc_weight_tensor_by_obsho[2][1, 4, GREAT, NORTH] = 4 * 10 ** 6

hc_weight_tensor_by_obsho[2][3, 4, GREAT, EAST] = 5 * 10 ** 6
hc_weight_tensor_by_obsho[2][3, 4, GREAT, SOUTH] = 4 * 10 ** 6

hc_weight_tensor_by_obsho[2][4, 3, GREAT, SOUTH] = 5 * 10 ** 6
hc_weight_tensor_by_obsho[2][4, 3, GREAT, EAST] = 4 * 10 ** 6

hc_weight_tensor_by_obsho[2][4, 1, GREAT, SOUTH] = 5 * 10 ** 6
hc_weight_tensor_by_obsho[2][4, 1, GREAT, WEST] = 4 * 10 ** 6

hc_weight_tensor_by_obsho[2][3, 0, GREAT, WEST] = 5 * 10 ** 6
hc_weight_tensor_by_obsho[2][3, 0, GREAT, SOUTH] = 4 * 10 ** 6

hc_weight_tensor_by_obsho[2][1, 0, GREAT, WEST] = 5 * 10 ** 6
hc_weight_tensor_by_obsho[2][1, 0, GREAT, NORTH] = 4 * 10 ** 6


hc_weight_tensor_by_obsho[2][0, 1, GOOD, NORTH] = 5 * 10 ** 3
hc_weight_tensor_by_obsho[2][0, 1, GOOD, WEST] = 4 * 10 ** 3

hc_weight_tensor_by_obsho[2][0, 3, GOOD, NORTH] = 5 * 10 ** 3
hc_weight_tensor_by_obsho[2][0, 3, GOOD, EAST] = 4 * 10 ** 3

hc_weight_tensor_by_obsho[2][1, 4, GOOD, EAST] = 5 * 10 ** 3
hc_weight_tensor_by_obsho[2][1, 4, GOOD, NORTH] = 4 * 10 ** 3

hc_weight_tensor_by_obsho[2][3, 4, GOOD, EAST] = 5 * 10 ** 3
hc_weight_tensor_by_obsho[2][3, 4, GOOD, SOUTH] = 4 * 10 ** 3

hc_weight_tensor_by_obsho[2][4, 3, GOOD, SOUTH] = 5 * 10 ** 3
hc_weight_tensor_by_obsho[2][4, 3, GOOD, EAST] = 4 * 10 ** 3

hc_weight_tensor_by_obsho[2][4, 1, GOOD, SOUTH] = 5 * 10 ** 3
hc_weight_tensor_by_obsho[2][4, 1, GOOD, WEST] = 4 * 10 ** 3

hc_weight_tensor_by_obsho[2][3, 0, GOOD, WEST] = 5 * 10 ** 3
hc_weight_tensor_by_obsho[2][3, 0, GOOD, SOUTH] = 4 * 10 ** 3

hc_weight_tensor_by_obsho[2][1, 0, GOOD, WEST] = 5 * 10 ** 3
hc_weight_tensor_by_obsho[2][1, 0, GOOD, NORTH] = 4 * 10 ** 3




hc_bias_tensor_by_obsho[2] = np.array([1, 0, -1, -2])
