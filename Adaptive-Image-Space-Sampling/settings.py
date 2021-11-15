import numpy as np

# The input file
FILE = "input/CT_head_64.png"
OUTPUT_FOLDER = "output/"

# Parameters for the rendering process
PERCENTAGE_PPI = 0.01
# NUM_NEIGHBOURS = 8
# POWER_PARAMETER = 3
NUM_NEAREST_NEIGHBOURS = np.array([4])
POWER_PARAMETERS = np.array([3])
POP_THRESHOLD = 25
POP_GROUP_SIZE = 3
GRID_RESOLUTION = (16, 16)

# Number of runs
NUM_RUNS = 2
# Number of CPUs
NUM_CPUS = 1

# For the popping visualization
COLOUR_MAP = 'viridis'
BACKGROUND_COLOUR = np.array([0, 0, 0, 0])  # RGBA


def title_info(n, p):
    return "(NN={:d}, p={:.2f}, T_pop={:.2f}, G_pop={:d})".format(n,
                                                                  p,
                                                                  POP_THRESHOLD,
                                                                  POP_GROUP_SIZE)


FPS = 20

# Graphics constants for the popping sketch
GRID_DIM = 64  # Set the image dimension manually
INTERVAL_SIZE = 6

DISTANCE_METRIC = 'cosine'

G_VERTICAL_SPACING = 50  # pixels
G_PADDING = 5  # pixels

G_COST_COLORMAP = 'plasma_r'
G_COST_COLORMAP_LENGTH = 256
G_COST_SYMBOL = 'o'
G_COST_PLACEMENT = 0.25  # [0, 1]: placement in VERTICAL_SPACING

G_POP_COLOR = 'black'
G_POP_SYMBOL = '^'
G_POP_PLACEMENT = 0.5  # [0, 1]: placement in VERTICAL_SPACING

G_LINE_COLOR = (0, 0, 0, 0.1)
G_LINE_PLACEMENT = 0.85  # [0, 1]: placement in VERTICAL_SPACING

G_X_TICKS_NUM = 11
