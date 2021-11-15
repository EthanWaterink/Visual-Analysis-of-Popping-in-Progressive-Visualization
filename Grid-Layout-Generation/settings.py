import numpy as np

# The input ZIP file
ZIP_FILE = "input/ethan_caltech1k_data.zip"
ZIP_FILE_PARAMETERS = "input/ethan_caltech_kstudy.zip"
OUTPUT_FOLDER = "output/"

# Collection
NUM_IMAGES = 1024
h = int(np.log(NUM_IMAGES) / np.log(4))
GRID_DIM = 2 ** h

# Process
NUM_RUNS = 16
NUM_SUB_ITERATIONS = 1024
NUM_SUPER_ITERATIONS = 1
NUM_TOTAL_ITERATIONS = NUM_SUPER_ITERATIONS * NUM_SUB_ITERATIONS
POP_THRESHOLD = .02
AT_OR_BELOW_H = 3
DISTANCE_METRIC = 'Hamming'
# Dividing the iterations into intervals
ITERATION_BLOCK_SIZE = 16
NUM_ITERATION_BLOCKS = np.ceil(NUM_TOTAL_ITERATIONS / ITERATION_BLOCK_SIZE).astype(np.int32)

# Graphics (G_) constants
G_NUM_FRAMES = 10
G_FRAMES = np.linspace(0, NUM_TOTAL_ITERATIONS - 1, G_NUM_FRAMES, dtype=np.int32)

G_PLOT_WIDTH = NUM_ITERATION_BLOCKS * GRID_DIM
G_PLOT_HEIGHT = (NUM_RUNS + 1) * GRID_DIM

G_VERTICAL_SPACING = 24  # pixels
G_PADDING = 5  # pixels

G_COST_COLORMAP = 'plasma'
G_COST_COLORMAP_LENGTH = 256
G_COST_SYMBOL = 'o'
G_COST_PLACEMENT = 0.25  # [0, 1]: placement in VERTICAL_SPACING

G_POP_COLOR = 'black'
G_POP_SYMBOL = '^'
G_POP_PLACEMENT = 0.5  # [0, 1]: placement in VERTICAL_SPACING

G_LINE_COLOR = (0, 0, 0, 0.1)
G_LINE_PLACEMENT = 0.85  # [0, 1]: placement in VERTICAL_SPACING

G_X_TICKS_NUM = 10

BACKGROUND_COLOUR = np.array([0, 0, 0, 0])  # RGBA
