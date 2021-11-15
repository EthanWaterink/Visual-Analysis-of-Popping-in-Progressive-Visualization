# Project
from settings import *
# Scientific
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
from sklearn import metrics
from scipy.spatial.distance import pdist, squareform, cosine
import scipy.cluster.hierarchy as sch
# Graphics
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib import animation


def visualize_results(data_normal, data_parameter_study):
    """
    Show the intermediate results as well as the visualizations of the improvements and changes of the grid layout
    generation :return:
    """
    visualize_refinement_characteristics_cost(data_normal["costs"])
    visualize_multi_run_overview(data_normal["costs"], data_normal["popping"], data_normal["small_images"])
    visualize_parameter_space(data_parameter_study["costs"], data_parameter_study["popping"], data_parameter_study["computation_time"])

    plt.show()


def visualize_refinement_characteristics_cost(costs):
    """
    Visualize the costs
    :return:
    """
    print("visualize_refinement_characteristics_cost")

    # Plot the costs
    fig, ax = plt.subplots()
    plt.rcParams.update({'font.size': 22})
    ax.plot(range(NUM_SUPER_ITERATIONS * NUM_SUB_ITERATIONS + 1), costs.T)
    # Labels
    ax.set_title("Cost progression")
    ax.set_ylabel("cost", fontsize=22)
    ax.set_xlabel("iteration", fontsize=22)
    plt.vlines(
        np.arange(1, NUM_SUPER_ITERATIONS) * (NUM_SUB_ITERATIONS + 1) - 0.5,
        np.min(costs),
        np.max(costs),
        colors=G_LINE_COLOR
    )
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.tick_params(axis='both', which='minor', labelsize=18)
    plt.savefig(OUTPUT_FOLDER + "cost_normal.png", bbox_inches='tight', dpi=500)


def visualize_multi_run_overview(costs, popping, small_images):
    """
    Visualize the popping artifacts
    :param popping: The popping array
    :param small_images: The small images
    :return:
    """
    print("visualize_multi_run_overview")

    # Compute the distance matrix
    dist = pdist(popping, metric=DISTANCE_METRIC)
    # Perform hierarchical clustering and return the list of node ids (the row permutation)
    perm = sch.leaves_list(sch.linkage(dist))

    # Initialize an array to aggregate the popping and metric values
    agg_pop = np.zeros(NUM_TOTAL_ITERATIONS, dtype='bool')
    agg_metric = np.zeros(NUM_TOTAL_ITERATIONS + 1, dtype=np.float64)

    # Create a new figure
    fig, ax = plt.subplots()
    plt.rcParams.update({'font.size': 22})

    # Normalize the costs
    norm = plt.Normalize(np.min(costs), np.max(costs))

    # Loop over all runs
    for r in range(NUM_RUNS):
        agg_pop = np.logical_or(agg_pop, popping[perm[r]])
        agg_metric += costs[perm[r]]

        # Plot a horizontal line for each run, where its colour is obtained from applying a color map to the cost
        points = np.array([
            np.linspace(0, G_PLOT_WIDTH, NUM_TOTAL_ITERATIONS - 1),
            np.repeat((r + 1) * GRID_DIM + (r + G_COST_PLACEMENT) * G_VERTICAL_SPACING,
                      NUM_TOTAL_ITERATIONS - 1)
        ]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments, cmap=G_COST_COLORMAP, norm=norm, zorder=1)
        # Set the values used for colormapping
        lc.set_array(costs[perm[r]])
        lc.set_linewidth(5)
        ax.add_collection(lc)

        # Plot a symbol (just below the cost line) indicating popping
        pop_block_f = np.arange(NUM_TOTAL_ITERATIONS) / ITERATION_BLOCK_SIZE
        pop_X = np.floor((pop_block_f[popping[perm[r]]]) * GRID_DIM).astype(int)
        plt.scatter(
            pop_X,
            np.repeat((r + 1) * GRID_DIM + (r + G_POP_PLACEMENT) * G_VERTICAL_SPACING, len(pop_X)),
            marker=G_POP_SYMBOL,
            c=G_POP_COLOR,
            zorder=2
        )

    # Plot a horizontal line for each run, where its colour is obtained from applying a color map to the cost
    points = np.array([
        np.linspace(0, G_PLOT_WIDTH, NUM_TOTAL_ITERATIONS - 1),
        np.repeat((NUM_RUNS + 1) * GRID_DIM + (NUM_RUNS + G_COST_PLACEMENT) * G_VERTICAL_SPACING,
                  NUM_TOTAL_ITERATIONS - 1)
    ]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, cmap=G_COST_COLORMAP, norm=norm, zorder=1)
    # Set the values used for colormapping
    lc.set_array(agg_metric / NUM_RUNS)
    lc.set_linewidth(5)
    ax.add_collection(lc)

    # Plot a symbol (just below the cost line) indicating popping
    pop_block_f = np.arange(NUM_TOTAL_ITERATIONS) / ITERATION_BLOCK_SIZE
    pop_X = np.floor((pop_block_f[agg_pop]) * GRID_DIM).astype(int)
    ax.scatter(
        pop_X,
        np.repeat((NUM_RUNS + 1) * GRID_DIM + (NUM_RUNS + G_POP_PLACEMENT) * G_VERTICAL_SPACING,
                  len(pop_X)),
        marker=G_POP_SYMBOL,
        c=G_POP_COLOR,
        zorder=2
    )

    # Initialize the full image (to ITERATION_BLOCK+1)
    small_images_full = np.zeros((
        G_PLOT_HEIGHT + (NUM_RUNS + 1) * G_VERTICAL_SPACING,
        G_PLOT_WIDTH)
    ) + ITERATION_BLOCK_SIZE + 1
    small_images_full2 = np.zeros((
        G_PLOT_HEIGHT + (NUM_RUNS + 1) * G_VERTICAL_SPACING,
        G_PLOT_WIDTH)
    ) + ITERATION_BLOCK_SIZE + 1
    # Loop over each run/iteration_block
    for iteration_block in range(NUM_ITERATION_BLOCKS):
        count = 0.0
        for run in range(NUM_RUNS):
            # Skip if no popping occurred
            if small_images[perm[run], iteration_block, 0, 0] == ITERATION_BLOCK_SIZE + 1:
                continue
            if small_images_full2[
                NUM_RUNS * GRID_DIM + NUM_RUNS * G_VERTICAL_SPACING, iteration_block * GRID_DIM] == ITERATION_BLOCK_SIZE + 1:
                small_images_full2[
                (NUM_RUNS * GRID_DIM + NUM_RUNS * G_VERTICAL_SPACING):(
                        (NUM_RUNS + 1) * GRID_DIM + NUM_RUNS * G_VERTICAL_SPACING),
                iteration_block * GRID_DIM:(iteration_block + 1) * GRID_DIM
                ] = 0
            count += 1
            # Fill the area in small_image_full with the corresponding small_image and normalize the small image
            # to [0,1] locally and then to [0, ITERATION_BLOCK]
            small_images_full[
            (run * GRID_DIM + run * G_VERTICAL_SPACING):((run + 1) * GRID_DIM + run * G_VERTICAL_SPACING),
            iteration_block * GRID_DIM:(iteration_block + 1) * GRID_DIM
            ] = small_images[perm[run], iteration_block] / np.max(
                small_images[perm[run], iteration_block]) * ITERATION_BLOCK_SIZE
            small_images_full2[
            (NUM_RUNS * GRID_DIM + NUM_RUNS * G_VERTICAL_SPACING):(
                    (NUM_RUNS + 1) * GRID_DIM + NUM_RUNS * G_VERTICAL_SPACING),
            iteration_block * GRID_DIM:(iteration_block + 1) * GRID_DIM
            ] += small_images[perm[run], iteration_block] / np.max(
                small_images[perm[run], iteration_block]) * ITERATION_BLOCK_SIZE
        if count != 0.0:
            small_images_full2[
            (NUM_RUNS * GRID_DIM + NUM_RUNS * G_VERTICAL_SPACING):(
                    (NUM_RUNS + 1) * GRID_DIM + NUM_RUNS * G_VERTICAL_SPACING),
            iteration_block * GRID_DIM:(iteration_block + 1) * GRID_DIM
            ] /= count

    # Get the specified colour map
    cmap = plt.get_cmap('viridis', 256 + 1)
    # Extract the colours
    cmap_colours = cmap(np.linspace(0, 1, ITERATION_BLOCK_SIZE + 1))
    # Add background colour to the list of colours
    cmap_colours = np.vstack((cmap_colours[::-1], [0, 0, 0, 0]))
    # Crate the new colour map
    cmap = ListedColormap(cmap_colours)
    # Plot the popping images
    plt.imshow(small_images_full, vmin=0, vmax=ITERATION_BLOCK_SIZE + 1, cmap=cmap)

    # Get the specified colour map
    cmap = plt.get_cmap('magma', 256 + 1)
    # Extract the colours
    cmap_colours = cmap(np.linspace(0, 1, ITERATION_BLOCK_SIZE + 1))
    # Add background colour to the list of colours
    cmap_colours = np.vstack((cmap_colours[::-1], [0, 0, 0, 0]))
    # Crate the new colour map
    cmap = ListedColormap(cmap_colours)
    # Plot the popping images
    plt.imshow(small_images_full2, vmin=0, vmax=ITERATION_BLOCK_SIZE + 1, cmap=cmap)

    # Add lines separating the grid blocks images
    plt.hlines(
        np.arange(1, NUM_RUNS) * GRID_DIM + (np.arange(NUM_RUNS - 1) + G_LINE_PLACEMENT) * G_VERTICAL_SPACING,
        -G_PADDING,
        G_PLOT_WIDTH - 0.5 + G_PADDING,
        colors=G_LINE_COLOR
    )
    plt.vlines(
        np.arange(1, NUM_ITERATION_BLOCKS) * GRID_DIM - 0.5,
        -G_PADDING,
        G_PLOT_HEIGHT + (NUM_RUNS + 1) * G_VERTICAL_SPACING - 0.5,
        colors=G_LINE_COLOR
    )

    plt.hlines(
        [NUM_RUNS * GRID_DIM + (NUM_RUNS - 1 + G_LINE_PLACEMENT - 0.05) * G_VERTICAL_SPACING],
        -G_PADDING,
        G_PLOT_WIDTH - 0.5 + G_PADDING,
        colors='k',
        linestyles='dashed',
        linewidth=1
    )

    # Labels and formatting
    plt.title("Popping\n(threshold = {:.2f}, metric = {:s})".format(POP_THRESHOLD, DISTANCE_METRIC))
    plt.ylabel("run")
    plt.xlabel("iteration")
    yyt = (perm + 1).tolist()
    yyt.append("agg")
    plt.yticks(
        (np.arange(NUM_RUNS + 1) + 0.5) * GRID_DIM + np.arange(NUM_RUNS + 1) * G_VERTICAL_SPACING,
        yyt
    )
    plt.xticks(
        np.linspace(0, 1, G_X_TICKS_NUM) * G_PLOT_WIDTH,
        np.linspace(0, NUM_TOTAL_ITERATIONS, G_X_TICKS_NUM, dtype=np.int32)
    )
    plt.ylim([G_PLOT_HEIGHT + (NUM_RUNS + 1) * G_VERTICAL_SPACING, -G_PADDING])
    plt.xlim([-G_PADDING, G_PLOT_WIDTH + G_PADDING])
    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER + "popping_threshold={:.2f}.png".format(POP_THRESHOLD), bbox_inches="tight", dpi=500)


def visualize_parameter_space(costs, popping, computation_time):
    """
    Parameter space analysis
    :return: A 2D array of the costs
    """
    print("visualize_parameter_space")

    plt.figure()
    plt.rcParams.update({'font.size': 22})
    plt.plot(costs[4 * 16:4 * 16 + 11].T, 'tab:purple')
    plt.plot(costs[3 * 16:4 * 16].T, 'tab:red')
    plt.plot(costs[2 * 16:3 * 16].T, 'tab:green')
    plt.plot(costs[1 * 16:2 * 16].T, 'tab:orange')
    plt.plot(costs[0 * 16:1 * 16].T, 'tab:blue')
    plt.title("Cost (parameter space)")
    plt.ylabel("cost")
    plt.xlabel("step")

    custom_lines = [
        Line2D([0], [0], color='tab:blue', lw=4),
        Line2D([0], [0], color='tab:orange', lw=4),
        Line2D([0], [0], color='tab:green', lw=4),
        Line2D([0], [0], color='tab:red', lw=4),
        Line2D([0], [0], color='tab:purple', lw=4),
    ]
    plt.legend(custom_lines, ['k = 2', 'k = 5', 'k = 10', 'k = 20', 'k = 40'])

    plt.figure()
    plt.rcParams.update({'font.size': 22})
    plt.plot(computation_time[4 * 16:4 * 16 + 11].T, 'tab:purple')
    plt.plot(computation_time[3 * 16:4 * 16].T, 'tab:red')
    plt.plot(computation_time[2 * 16:3 * 16].T, 'tab:green')
    plt.plot(computation_time[1 * 16:2 * 16].T, 'tab:orange')
    plt.plot(computation_time[0 * 16:1 * 16].T, 'tab:blue')
    plt.title("Computation time (parameter space)")
    plt.ylabel("time")
    plt.xlabel("step")
    plt.legend(custom_lines, ['k = 2', 'k = 5', 'k = 10', 'k = 20', 'k = 40'])

    # Get the specified colour map
    cmap = plt.get_cmap('viridis', 2 * NUM_TOTAL_ITERATIONS)
    # Extract the colours
    cmap_colours = cmap(np.linspace(0, 1, 2 * NUM_TOTAL_ITERATIONS))
    # Add background colour to the list of colours
    cmap_colours = np.vstack((BACKGROUND_COLOUR, cmap_colours[::-1]))
    # Set transparency of the colours
    cmap_colours[1:, 3] = 0.35
    # Crate the new colour map
    cmap = ListedColormap(cmap_colours)

    # Set values
    block_size = 36
    block_spacing = 2
    num_rows = 3
    num_cols = 5

    # Plot image with popping distribution
    fig = plt.figure()
    ax = plt.subplot(111)
    plt.rcParams.update({'font.size': 22})

    # Create arrays for the aggregate row/column
    popping_row_agg = np.zeros((num_cols,), dtype=object)
    for i in range(num_cols):
        popping_row_agg[i] = []
    popping_col_agg = np.zeros((num_rows,), dtype=object)
    for i in range(num_rows):
        popping_col_agg[i] = []
    popping_agg_agg = []

    # Initialize an array for the median popping
    median_pop = np.zeros((block_size * (num_rows + 1) + (num_rows + 1 + 2) * block_spacing, block_size * (num_cols + 1)), np.float64)
    # Loop over all grid cells
    for k in range(num_cols):
        for t in range(num_rows):
            median_step = 0
            # If popping artifacts occurred in this block, compute the median
            if popping[k][t][0]:
                median_step = np.median(popping[k][t][0])

            # Add popping to respective aggregate array
            popping_row_agg[k] += popping[k][t][0]
            popping_col_agg[t] += popping[k][t][0]
            popping_agg_agg += popping[k][t][0]

            # Determine the value for the (color) bands
            median_pop[
            (t * block_size + 2 * block_size // 3 + (t + 1) * block_spacing):(t * block_size + block_size + (t + 1) * block_spacing),
            k * block_size:(k + 1) * block_size
            ] = np.percentile(popping[k][t][0], 0)
            median_pop[
            (t * block_size + block_size // 3 + (t + 1) * block_spacing):(t * block_size + 2 * block_size // 3 + (t + 1) * block_spacing),
            k * block_size:(k + 1) * block_size
            ] = median_step
            median_pop[
            (t * block_size + 0 + (t + 1) * block_spacing):(t * block_size + block_size // 3 + (t + 1) * block_spacing),
            k * block_size:(k + 1) * block_size
            ] = np.percentile(popping[k][t][0], 100)

    # Fll the row aggregate
    for k in range(num_cols):
        median_step = 0
        # If popping artifacts occurred in this block, compute the median
        if popping_row_agg[k]:
            median_step = np.median(popping_row_agg[k])

        # Determine the value for the (color) bands
        median_pop[
        (num_rows * block_size + 2 * block_size // 3 + (num_rows + 1) * block_spacing):(num_rows * block_size + block_size + (num_rows + 1) * block_spacing),
        k * block_size:(k + 1) * block_size
        ] = np.percentile(popping_row_agg[k], 0)
        median_pop[
        (num_rows * block_size + block_size // 3 + (num_rows + 1) * block_spacing):(num_rows * block_size + 2 * block_size // 3 + (num_rows + 1) * block_spacing),
        k * block_size:(k + 1) * block_size
        ] = median_step
        median_pop[
        (num_rows * block_size + 0 + (num_rows + 1) * block_spacing):(num_rows * block_size + block_size // 3 + (num_rows + 1) * block_spacing),
        k * block_size:(k + 1) * block_size
        ] = np.percentile(popping_row_agg[k], 100)

    # Fll the column aggregate
    for t in range(num_rows):
        median_step = 0
        # If popping artifacts occurred in this block, compute the median
        if popping_col_agg[t]:
            median_step = np.median(popping_col_agg[t])

        # Determine the value for the (color) bands
        median_pop[
        (t * block_size + 2 * block_size // 3 + (t + 1) * block_spacing):(t * block_size + block_size + (t + 1) * block_spacing),
        num_cols * block_size:(num_cols + 1) * block_size
        ] = np.percentile(popping_col_agg[t], 0)
        median_pop[
        (t * block_size + block_size // 3 + (t + 1) * block_spacing):(t * block_size + 2 * block_size // 3 + (t + 1) * block_spacing),
        num_cols * block_size:(num_cols + 1) * block_size
        ] = median_step
        median_pop[
        (t * block_size + 0 + (t + 1) * block_spacing):(t * block_size + block_size // 3 + (t + 1) * block_spacing),
        num_cols * block_size:(num_cols + 1) * block_size
        ] = np.percentile(popping_col_agg[t], 100)

    # Fll the overall aggregate
    median_step = 0
    # If popping artifacts occurred in this block, compute the median
    if popping_agg_agg:
        median_step = np.median(popping_agg_agg)

    # Determine the value for the (color) bands
    median_pop[
    (num_rows * block_size + 2 * block_size // 3 + (num_rows + 1) * block_spacing):(num_rows * block_size + block_size + (num_rows + 1) * block_spacing),
    num_cols * block_size:(num_cols + 1) * block_size
    ] = np.percentile(popping_agg_agg, 0)
    median_pop[
    (num_rows * block_size + block_size // 3 + (num_rows + 1) * block_spacing):(num_rows * block_size + 2 * block_size // 3 + (num_rows + 1) * block_spacing),
    num_cols * block_size:(num_cols + 1) * block_size
    ] = median_step
    median_pop[
    (num_rows * block_size + 0 + (num_rows + 1) * block_spacing):(num_rows * block_size + block_size // 3 + (num_rows + 1) * block_spacing),
    num_cols * block_size:(num_cols + 1) * block_size
    ] = np.percentile(popping_agg_agg, 100)

    # Show the median popping array
    im = ax.imshow(median_pop, vmin=0, vmax=NUM_TOTAL_ITERATIONS, cmap=cmap)
    # Add a color bar
    fig.colorbar(im, label="step")
    # Remove the frame
    plt.setp(ax, 'frame_on', False)
    # Loop over all grid cells
    for k in np.arange(num_cols + 1):
        for t in np.arange(num_rows + 1):
            if k < num_cols and t < num_rows:
                pppp = popping[k][t][0]
            elif k == num_cols and t == num_rows:
                pppp = popping_agg_agg
            elif t == num_rows:
                pppp = popping_row_agg[k]
            else:
                pppp = popping_col_agg[t]

            # Show a boxplot for the popping time step distribution inside this grid cell. To make sure the boxplot
            # fits inside the grid cell, we first scale the popping time steps to [0,1]. Then, the origin of the
            # image is 'up', we invert the time steps. Then, we add the grid cell's height to the vertical position
            # of the boxplot. Its horizontal position is at the center of the grid cell, and its width is 75% of the
            # grid cell's width.
            ax.boxplot(
                (1 - np.array(pppp) / (NUM_TOTAL_ITERATIONS - 2) + t) * block_size - 0.5 + (t + 1) * block_spacing,
                positions=([(k + 0.5) * block_size]),
                widths=0.75 * block_size)
    ax.set_title("Popping distribution parameter space")
    plt.xticks((np.arange(num_cols + 1) + 0.5) * block_size, [2, 5, 10, 20, 40, "agg(k)"])
    plt.yticks((np.arange(num_rows + 1) + 0.5) * block_size + block_spacing + np.arange(num_rows + 1) * block_spacing - 0.5,
               [.005, .01, .02, "agg(T_pop)"])
    ax.set_ylabel("T_pop", fontsize=22)
    ax.set_xlabel("k", fontsize=22)
    plt.hlines(np.arange(num_rows, num_rows + 1) * block_size + (num_rows + 0.5) * block_spacing - 0.5, -0.5, (num_cols + 1) * block_size - 0.5,
               colors='k', linestyles='dashed', linewidth=1)
    plt.vlines(np.arange(1, num_cols) * block_size - 0.5, -0.5, (num_rows + 1) * block_size + (num_rows + 2) * block_spacing - 0.5, colors='k',
               linewidth=1)
    plt.vlines(np.arange(num_cols, num_cols + 1) * block_size - 0.5, -0.5, (num_rows + 1) * block_size + (num_rows + 2) * block_spacing - 0.5, colors='k',
               linestyles='dashed',
               linewidth=1)
    plt.grid(False)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.tick_params(axis='both', which='minor', labelsize=18)
    plt.rc('axes', labelsize=22)  # fontsize of the x and y labels

    plt.savefig("output/popping_parameter.png", bbox_inches='tight', dpi=500)
