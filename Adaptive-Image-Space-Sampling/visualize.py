# Scientific
import numpy as np
from scipy.spatial.distance import pdist, squareform, cosine
import scipy.cluster.hierarchy as sch
# Misc
import time
# Graphics
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib import animation
from skimage import color
# Project
from FakeRenderer import FakeRenderer
from settings import *


def visualize_results(fake_renderers):
    """
    Show the intermediate results as well as the visualizations of the improvements and changes of the fake renderer
    """
    # Start timer
    print("=== Compute visualizations ===")
    start = time.time()

    # Visualize the results (of multiple runs and a single run (e.g. the first one))
    visualize_refinement_characteristics_similarity(fake_renderers)
    visualize_refinement_characteristics_pixel_difference(fake_renderers[0])
    visualize_multi_run_overview(fake_renderers)
    visualize_positional_analysis_sampling_time(fake_renderers[0])
    visualize_positional_analysis_popping(fake_renderers[0])
    visualize_parameter_space(fake_renderers)

    # Lap timer end print total time for computing the visualizations
    print("Total time: {:.3f}".format(time.time() - start))

    # Animate the results of a single run (e.g. the first one)
    # animate_reconstruction(fake_renderers[0])
    # animate_refinement_characteristics_similarity(fake_renderers[0])
    # animate_positional_analysis_sampling_time(fake_renderers[0])
    # animate_positional_analysis_popping(fake_renderers[0])

    # Stop timer end print total time for computing the animations
    print("Total time after animation: {:.3f}".format(time.time() - start))

    plt.show()


def colour_map(fake_renderer, transparency):
    """
    Return a colour map with a background colour (see Settings) where the colours may be transparent
    :param fake_renderer: The fake renderer
    :param transparency: The transparency
    :return:
    """
    # Create a colour map that shows some background colour for pixels that are
    # not sampled (which have value 0), and use some real colour map
    # values [1, num_steps]
    #
    # Get the specified colour map
    cmap = plt.get_cmap(COLOUR_MAP, 2 * fake_renderer.NUM_ITERATIONS)
    # Extract the colours
    cmap_colours = cmap(np.linspace(0, 1, fake_renderer.NUM_ITERATIONS))
    # Add background colour to the list of colours
    cmap_colours = np.vstack((BACKGROUND_COLOUR, cmap_colours[::-1]))
    # Set transparency of the colours
    cmap_colours[1:, 3] = transparency
    # Crate the new colour map
    return ListedColormap(cmap_colours)


# ====================================== Visualization (plots) ===============================================

def visualize_refinement_characteristics_similarity(fake_renderers):
    """
    Visualize the similarity by plotting it
    :return:
    """
    print("visualize_refinement_characteristics_similarity")

    fig, ax = plt.subplots()
    for fr in fake_renderers:
        # Plot the similarity versus the iterations
        ax.plot(
            np.arange(2, fr.NUM_ITERATIONS + 1),
            fr.similarity,
        )
        # Plot the similarity (compared to the final image) versus the iterations
        ax.plot(
            np.arange(2, fr.NUM_ITERATIONS + 1),
            fr.similarity_final,
        )
    # Labels
    ax.set_title("Reconstructed image similarity\n{:s}".format(title_info(fake_renderers[0].NUM_NEIGHBOURS, fake_renderers[0].POWER_PARAMETER)))
    ax.set_ylabel("SSIM")
    ax.set_xlabel("step")
    ax.legend(["Consecutive", "Final"])
    plt.rcParams.update({'font.size': 22})
    plt.savefig(OUTPUT_FOLDER + "similarity.png", bbox_inches="tight", dpi=500)


def visualize_refinement_characteristics_pixel_difference(fake_renderer):
    """
    Visualize the signal by plotting it
    :param fake_renderer: The fake renderer
    :return:
    """
    print("visualize_refinement_characteristics_pixel_difference")

    # Signal
    fig, ax = plt.subplots()
    # Loop over each row and plot the signal for all pixels in that row
    for r in range(fake_renderer.RESOLUTION[0]):
        ax.plot(np.arange(1, fake_renderer.NUM_ITERATIONS + 1) + 1, np.transpose(fake_renderer.signal[r, :]))
    # Plot the popping threshold as a horizontal line
    ax.plot([2, fake_renderer.NUM_ITERATIONS + 1], [POP_THRESHOLD, POP_THRESHOLD], 'k', linewidth=2)
    # Labels
    ax.set_title('(Reconstructed) pixel difference\n{:s}'.format(title_info(fake_renderer.NUM_NEIGHBOURS, fake_renderer.POWER_PARAMETER)))
    ax.set_ylabel('Delta E*')
    ax.set_xlabel('Step')
    plt.rcParams.update({'font.size': 22})
    plt.savefig(OUTPUT_FOLDER + "signal.png", bbox_inches="tight", dpi=500)


def visualize_positional_analysis_sampling_time(fake_renderer):
    """
    Visualize when pixels were sampled by showing a 2D image using a colour map
    :param fake_renderer: The fake renderer
    :return:
    """
    print("visualize_positional_analysis_sampling_time")

    # Plot scheme.when
    fig, ax = plt.subplots()
    plt.rcParams.update({'font.size': 22})
    ax.axis('off')
    im = ax.imshow(fake_renderer.when, cmap=colour_map(fake_renderer, 1), vmin=0, vmax=fake_renderer.NUM_ITERATIONS,
                   interpolation=None)
    fig.colorbar(im, label="step")
    # Labels
    ax.set_title('W_n\n{:s}'.format(title_info(fake_renderer.NUM_NEIGHBOURS, fake_renderer.POWER_PARAMETER)))
    plt.savefig(OUTPUT_FOLDER + "when.png", bbox_inches="tight", dpi=500)


def visualize_multi_run_overview(fake_renderers):
    """
    Visualize the results of several runs by stacking them and showing the following:
    - A metric bar showing the value of some metric
    - Triangles indicating time steps at which popping occurred
    - Boxes of spatial changes showing where significant changes happened
    - The runs are reordered to improve visual clarity
    - An additional row aggregates all the runs to summarize them
    :param fake_renderers: A list of fake renderers
    :return:
    """
    print("visualize_multi_run_overview")

    # Create a 2D array of the popping
    popping = np.zeros((NUM_RUNS, fake_renderers[0].NUM_ITERATIONS - 1))
    for r in range(NUM_RUNS):
        popping[r] = fake_renderers[r].popping_arr
    # Compute the distance matrix
    dist = pdist(popping, metric=DISTANCE_METRIC)
    # Perform hierarchical clustering and return the list of node ids (the row permutation)
    perm = sch.leaves_list(sch.linkage(dist))

    # Create a 2D array of the similarities
    similarities = np.zeros((NUM_RUNS, fake_renderers[0].NUM_ITERATIONS - 1))
    for r in range(NUM_RUNS):
        similarities[r] = fake_renderers[r].similarity_final

    # Determine the plot dimensions
    G_PLOT_WIDTH = fake_renderers[0].NUM_INTERVALS * GRID_DIM
    G_PLOT_HEIGHT = (NUM_RUNS+1) * GRID_DIM

    # Create a new figure
    fig, ax = plt.subplots()
    plt.rcParams.update({'font.size': 22})

    # Initialize an array to aggregate the popping and metric values
    agg_pop = np.zeros(fake_renderers[0].NUM_ITERATIONS-1, dtype='bool')
    agg_metric = np.zeros(fake_renderers[0].NUM_ITERATIONS-1, dtype=np.float64)

    # Normalize the similarities
    norm = plt.Normalize(np.min(similarities), np.max(similarities))

    # Loop over all runs
    for r in range(NUM_RUNS):
        agg_pop = np.logical_or(agg_pop, fake_renderers[perm[r]].popping_arr)
        agg_metric += similarities[perm[r]]

        # Plot a horizontal line for each run, where its colour is obtained from applying a color map to the cost
        points = np.array([
            np.linspace(0, G_PLOT_WIDTH, fake_renderers[0].NUM_ITERATIONS - 1),
            np.repeat((r + 1) * GRID_DIM + (r + G_COST_PLACEMENT) * G_VERTICAL_SPACING,
                      fake_renderers[0].NUM_ITERATIONS - 1)
        ]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments, cmap=G_COST_COLORMAP, norm=norm, zorder=1)
        # Set the values used for colormapping
        lc.set_array(similarities[perm[r]])
        lc.set_linewidth(5)
        ax.add_collection(lc)

        # Plot a symbol (just below the cost line) indicating popping
        pop_block_f = np.arange(fake_renderers[0].NUM_ITERATIONS - 1) / INTERVAL_SIZE
        pop_X = np.floor((pop_block_f[fake_renderers[perm[r]].popping_arr]) * GRID_DIM).astype(int)
        ax.scatter(
            pop_X,
            np.repeat((r + 1) * GRID_DIM + (r + G_POP_PLACEMENT) * G_VERTICAL_SPACING, len(pop_X)),
            marker=G_POP_SYMBOL,
            c=G_POP_COLOR,
            zorder=2
        )

    # Plot a horizontal line for each run, where its colour is obtained from applying a color map to the cost
    points = np.array([
        np.linspace(0, G_PLOT_WIDTH, fake_renderers[0].NUM_ITERATIONS - 1),
        np.repeat((NUM_RUNS + 1) * GRID_DIM + (NUM_RUNS + G_COST_PLACEMENT) * G_VERTICAL_SPACING,
                  fake_renderers[0].NUM_ITERATIONS - 1)
    ]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, cmap=G_COST_COLORMAP, norm=norm, zorder=1)
    # Set the values used for color mapping
    lc.set_array(agg_metric/NUM_RUNS)
    lc.set_linewidth(5)
    ax.add_collection(lc)

    # Plot a symbol (just below the cost line) indicating popping
    pop_block_f = np.arange(fake_renderers[0].NUM_ITERATIONS - 1) / INTERVAL_SIZE
    pop_X = np.floor((pop_block_f[agg_pop]) * GRID_DIM).astype(int)
    ax.scatter(
        pop_X,
        np.repeat((NUM_RUNS+1) * GRID_DIM + (NUM_RUNS + G_POP_PLACEMENT) * G_VERTICAL_SPACING, len(pop_X)),
        marker=G_POP_SYMBOL,
        c=G_POP_COLOR,
        zorder=2
    )
    sm = plt.cm.ScalarMappable(norm=norm, cmap=G_COST_COLORMAP)
    sm.set_array([])
    fig.colorbar(sm, ticks=np.linspace(np.min(similarities), np.max(similarities), 11), label="similarity", fraction=G_PLOT_HEIGHT/G_PLOT_WIDTH*0.055)

    # Initialize the full image (to ITERATION_BLOCK+1)
    small_images_full = np.zeros((
        G_PLOT_HEIGHT + (NUM_RUNS+1) * G_VERTICAL_SPACING,
        G_PLOT_WIDTH)
    ) + INTERVAL_SIZE + 1
    small_images_full2 = np.zeros((
        G_PLOT_HEIGHT + (NUM_RUNS+1) * G_VERTICAL_SPACING,
        G_PLOT_WIDTH)
    ) + INTERVAL_SIZE + 1
    # Loop over each run/iteration_block
    for iteration_block in range(fake_renderers[0].NUM_INTERVALS):
        count = 0.0
        for run in range(NUM_RUNS):
            # Skip if no popping occurred
            if fake_renderers[perm[run]].small_images[iteration_block, 0, 0] == INTERVAL_SIZE + 1:
                continue
            if small_images_full2[NUM_RUNS * GRID_DIM + NUM_RUNS * G_VERTICAL_SPACING,iteration_block * GRID_DIM] == INTERVAL_SIZE+1:
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
            ] = fake_renderers[perm[run]].small_images[iteration_block] / np.max(
                fake_renderers[perm[run]].small_images[iteration_block]) * INTERVAL_SIZE
            small_images_full2[
            (NUM_RUNS * GRID_DIM + NUM_RUNS * G_VERTICAL_SPACING):((NUM_RUNS + 1) * GRID_DIM + NUM_RUNS * G_VERTICAL_SPACING),
            iteration_block * GRID_DIM:(iteration_block + 1) * GRID_DIM
            ] += fake_renderers[perm[run]].small_images[iteration_block]
        if count != 0.0:
            small_images_full2[
            (NUM_RUNS * GRID_DIM + NUM_RUNS * G_VERTICAL_SPACING):(
                        (NUM_RUNS + 1) * GRID_DIM + NUM_RUNS * G_VERTICAL_SPACING),
            iteration_block * GRID_DIM:(iteration_block + 1) * GRID_DIM
            ] /= count
            small_images_full2[
            (NUM_RUNS * GRID_DIM + NUM_RUNS * G_VERTICAL_SPACING):(
                    (NUM_RUNS + 1) * GRID_DIM + NUM_RUNS * G_VERTICAL_SPACING),
            iteration_block * GRID_DIM:(iteration_block + 1) * GRID_DIM
            ] /= np.max(small_images_full2[
            (NUM_RUNS * GRID_DIM + NUM_RUNS * G_VERTICAL_SPACING):(
                        (NUM_RUNS + 1) * GRID_DIM + NUM_RUNS * G_VERTICAL_SPACING),
            iteration_block * GRID_DIM:(iteration_block + 1) * GRID_DIM
            ])
            small_images_full2[
            (NUM_RUNS * GRID_DIM + NUM_RUNS * G_VERTICAL_SPACING):(
                    (NUM_RUNS + 1) * GRID_DIM + NUM_RUNS * G_VERTICAL_SPACING),
            iteration_block * GRID_DIM:(iteration_block + 1) * GRID_DIM
            ] *= INTERVAL_SIZE

    # Get the specified colour map
    cmap = plt.get_cmap('viridis', 256 + 1)
    # Extract the colours
    cmap_colours = cmap(np.linspace(0, 1, INTERVAL_SIZE + 1))
    # Add background colour to the list of colours
    cmap_colours = np.vstack((cmap_colours[::-1], [0, 0, 0, 0]))
    # Crate the new colour map
    cmap = ListedColormap(cmap_colours)
    # Plot the popping images
    plt.imshow(small_images_full, vmin=0, vmax=INTERVAL_SIZE + 1, cmap=cmap)

    # Get the specified colour map
    cmap = plt.get_cmap('magma', 256 + 1)
    # Extract the colours
    cmap_colours = cmap(np.linspace(0, 1, INTERVAL_SIZE + 1))
    # Add background colour to the list of colours
    cmap_colours = np.vstack((cmap_colours[::-1], [0, 0, 0, 0]))
    # Crate the new colour map
    cmap = ListedColormap(cmap_colours)
    # Plot the popping images
    plt.imshow(small_images_full2, vmin=0, vmax=INTERVAL_SIZE + 1, cmap=cmap)

    # Add lines separating the grid blocks images
    plt.hlines(
        np.arange(1, NUM_RUNS) * GRID_DIM + (np.arange(NUM_RUNS - 1) + G_LINE_PLACEMENT) * G_VERTICAL_SPACING,
        -G_PADDING,
        G_PLOT_WIDTH - 0.5 + G_PADDING,
        colors=G_LINE_COLOR
    )
    plt.vlines(
        np.arange(1, fake_renderers[0].NUM_INTERVALS) * GRID_DIM - 0.5,
        -G_PADDING,
        G_PLOT_HEIGHT + (NUM_RUNS+1) * G_VERTICAL_SPACING - 0.5,
        colors=G_LINE_COLOR
    )

    plt.hlines(
        [NUM_RUNS * GRID_DIM + (NUM_RUNS-1 + G_LINE_PLACEMENT - 0.05) * G_VERTICAL_SPACING],
        -G_PADDING,
        G_PLOT_WIDTH - 0.5 + G_PADDING,
        colors='k',
        linestyles='dashed',
        linewidth=1
    )

    # Labels and formatting
    plt.title("Popping\n(threshold = {:.2f}, metric = {:s})".format(POP_THRESHOLD, DISTANCE_METRIC))
    plt.ylabel("run", fontsize=20)
    plt.xlabel("iteration", fontsize=20)
    yyt = (perm+1).tolist()
    yyt.append("agg")
    plt.yticks(
        (np.arange(NUM_RUNS+1) + 0.5) * GRID_DIM + np.arange(NUM_RUNS+1) * G_VERTICAL_SPACING,
        yyt
    )
    plt.xticks(
        np.linspace(0, 1, G_X_TICKS_NUM) * G_PLOT_WIDTH,
        np.linspace(1, fake_renderers[0].NUM_ITERATIONS, G_X_TICKS_NUM, dtype=np.int32)
    )
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.tick_params(axis='both', which='minor', labelsize=18)
    plt.ylim([G_PLOT_HEIGHT + (NUM_RUNS+1) * G_VERTICAL_SPACING, -G_PADDING])
    plt.xlim([-G_PADDING, G_PLOT_WIDTH + G_PADDING])
    plt.tight_layout()
    plt.savefig("output/popping_threshold={:.2f}.png".format(POP_THRESHOLD), bbox_inches="tight", dpi=500)


def visualize_parameter_space(fake_renderers):
    """
    Show the popping distribution per parameter combination. An additional row/column aggregates their respective
    column/row to summarize them. The bottom right block aggregates all distributions. :param fake_renderers: :return:
    """
    print("visualize_parameters")

    # Set values
    block_size = 36
    block_spacing = 2
    # Get values
    num_rows = NUM_NEAREST_NEIGHBOURS.shape[0]
    num_cols = POWER_PARAMETERS.shape[0]

    # Initialize an array to store the popping distributions per parameter combination
    popping_comb = np.zeros((num_rows, num_cols), dtype=object)
    for n in range(num_rows):
        for p in range(num_cols):
            popping_comb[n, p] = []
            for r in range(NUM_RUNS):
                arr = np.where(fake_renderers[n * num_cols * NUM_RUNS + p * NUM_RUNS + r].popping_arr)[0] + 1
                popping_comb[n, p] += arr.tolist()

    # Get the specified colour map
    cmap = plt.get_cmap(COLOUR_MAP, 2 * fake_renderers[0].NUM_ITERATIONS)
    # Extract the colours
    cmap_colours = cmap(np.linspace(0, 1, 2 * fake_renderers[0].NUM_ITERATIONS))
    # Add background colour to the list of colours
    cmap_colours = np.vstack((BACKGROUND_COLOUR, cmap_colours[::-1]))
    # Set transparency of the colours
    cmap_colours[1:, 3] = 0.35
    # Crate the new colour map
    cmap = ListedColormap(cmap_colours)

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
    # ... and an array for aggregating all results
    popping_agg_agg = []

    # Initialize an array for the median popping
    median_pop = np.zeros((block_size * (num_rows + 1) + (num_rows + 1 + 2) * block_spacing, block_size * (num_cols + 1)), np.float64)
    # Loop over all grid cells
    for r in range(num_rows):
        for c in range(num_cols):
            median_step = 0
            # If popping artifacts occurred in this block, compute the median
            if popping_comb[r, c]:
                median_step = np.median(popping_comb[r, c])

            # Add popping to respective aggregate array
            popping_row_agg[c] += popping_comb[r, c]
            popping_col_agg[r] += popping_comb[r, c]
            popping_agg_agg += popping_comb[r, c]

            # Determine the value for the (color) bands
            median_pop[
            (r * block_size + 2 * block_size // 3 + (r + 1) * block_spacing):(r * block_size + block_size + (r + 1) * block_spacing),
            c * block_size:(c + 1) * block_size
            ] = np.percentile(popping_comb[r, c], 0)
            median_pop[
            (r * block_size + block_size // 3 + (r + 1) * block_spacing):(r * block_size + 2 * block_size // 3 + (r + 1) * block_spacing),
            c * block_size:(c + 1) * block_size
            ] = median_step
            median_pop[
            (r * block_size + 0 + (r + 1) * block_spacing):(r * block_size + block_size // 3 + (r + 1) * block_spacing),
            c * block_size:(c + 1) * block_size
            ] = np.percentile(popping_comb[r, c], 100)

    # Fll the row aggregate
    for c in range(num_cols):
        median_step = 0
        # If popping artifacts occurred in this block, compute the median
        if popping_row_agg[c]:
            median_step = np.median(popping_row_agg[c])

        # Determine the value for the (color) bands
        median_pop[
        (num_rows * block_size + 2 * block_size // 3 + (num_rows + 1) * block_spacing):(num_rows * block_size + block_size + (num_rows + 1) * block_spacing),
        c * block_size:(c + 1) * block_size
        ] = np.percentile(popping_row_agg[c], 0)  # np.min(popping[k][t][0])
        median_pop[
        (num_rows * block_size + block_size // 3 + (num_rows + 1) * block_spacing):(num_rows * block_size + 2 * block_size // 3 + (num_rows + 1) * block_spacing),
        c * block_size:(c + 1) * block_size
        ] = median_step
        median_pop[
        (num_rows * block_size + 0 + (num_rows + 1) * block_spacing):(num_rows * block_size + block_size // 3 + (num_rows + 1) * block_spacing),
        c * block_size:(c + 1) * block_size
        ] = np.percentile(popping_row_agg[c], 100)

    # Fll the column aggregate
    for r in range(num_rows):
        median_step = 0
        # If popping artifacts occurred in this block, compute the median
        if popping_col_agg[r]:
            median_step = np.median(popping_col_agg[r])

        # Determine the value for the (color) bands
        median_pop[
        (r * block_size + 2 * block_size // 3 + (r + 1) * block_spacing):(r * block_size + block_size + (r + 1) * block_spacing),
        num_cols * block_size:(num_cols + 1) * block_size
        ] = np.percentile(popping_col_agg[r], 0)
        median_pop[
        (r * block_size + block_size // 3 + (r + 1) * block_spacing):(r * block_size + 2 * block_size // 3 + (r + 1) * block_spacing),
        num_cols * block_size:(num_cols + 1) * block_size
        ] = median_step
        median_pop[
        (r * block_size + 0 + (r + 1) * block_spacing):(r * block_size + block_size // 3 + (r + 1) * block_spacing),
        num_cols * block_size:(num_cols + 1) * block_size
        ] = np.percentile(popping_col_agg[r], 100)

    # Fll the overall aggregate
    median_step = 0
    # If popping artifacts occurred in this block, compute the median
    if popping_agg_agg:
        median_step = np.median(popping_agg_agg)

    # Determine the value for the (color) bands
    median_pop[
    (num_rows * block_size + 2 * block_size // 3 + (num_rows + 1) * block_spacing):(num_rows * block_size + block_size + (num_rows + 1) * block_spacing),
    num_cols * block_size:(num_cols + 1) * block_size
    ] = np.percentile(popping_agg_agg, 0)  # np.min(popping[k][t][0])
    median_pop[
    (num_rows * block_size + block_size // 3 + (num_rows + 1) * block_spacing):(num_rows * block_size + 2 * block_size // 3 + (num_rows + 1) * block_spacing),
    num_cols * block_size:(num_cols + 1) * block_size
    ] = median_step
    median_pop[
    (num_rows * block_size + 0 + (num_rows + 1) * block_spacing):(num_rows * block_size + block_size // 3 + (num_rows + 1) * block_spacing),
    num_cols * block_size:(num_cols + 1) * block_size
    ] = np.percentile(popping_agg_agg, 100)  # np.max(popping[k][t][0])

    # Show the median popping array
    im = ax.imshow(median_pop, vmin=0, vmax=fake_renderers[0].NUM_ITERATIONS, cmap=cmap)
    # Add a color bar
    fig.colorbar(im, label="step")
    # Remove the frame
    plt.setp(ax, 'frame_on', False)
    # Loop over all grid cells
    for r in np.arange(num_rows + 1):
        for c in np.arange(num_cols + 1):
            if c < num_cols and r < num_rows:
                pppp = popping_comb[r, c]
            elif c == num_cols and r == num_rows:
                pppp = popping_agg_agg
            elif r == num_rows:
                pppp = popping_row_agg[c]
            else:
                pppp = popping_col_agg[r]
            # Show a boxplot for the popping time step distribution inside this grid cell. To make sure the boxplot
            # fits inside the grid cell, we first scale the popping time steps to [0,1]. Then, the origin of the
            # image is 'up', we invert the time steps. Then, we add the grid cell's height to the vertical position
            # of the boxplot. Its horizontal position is at the center of the grid cell, and its width is 75% of the
            # grid cell's width.
            ax.boxplot(
                (1 - np.array(pppp) / fake_renderers[0].NUM_ITERATIONS + r) * block_size - 0.5 + (r + 1) * block_spacing,
                positions=([(c + 0.5) * block_size]),
                widths=0.75 * block_size)
    ax.set_title(
        "Popping distribution parameter space\n(T_pop={:.2f}, G_pop={:.2f})".format(POP_THRESHOLD, POP_GROUP_SIZE))
    plt.yticks((np.arange(num_rows + 1) + 0.5) * block_size + block_spacing + np.arange(num_rows + 1) * block_spacing, NUM_NEAREST_NEIGHBOURS.tolist() + ["agg(NN)"])
    plt.xticks((np.arange(num_cols + 1) + 0.5) * block_size, POWER_PARAMETERS.tolist() + ["agg(PP)"])
    plt.ylabel("number of nearest neighbours", fontsize=22)
    plt.xlabel("power parameter", fontsize=22)
    plt.hlines(np.arange(num_rows, num_rows + 1) * block_size + (num_rows + 0.5) * block_spacing - 0.5, -0.5, (num_cols + 1) * block_size - 0.5, colors='k',
               linestyles='dashed', linewidth=1)
    plt.vlines(np.arange(1, num_cols) * block_size - 0.5, -0.5, (num_rows + 1) * block_size + (num_rows + 2) * block_spacing - 0.5, colors='k',
               linewidth=1)
    plt.vlines(np.arange(num_cols, num_cols + 1) * block_size - 0.5, -0.5, (num_rows + 1) * block_size + (num_rows + 2) * block_spacing - 0.5, colors='k',
               linestyles='dashed', linewidth=1)
    plt.grid(False)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.tick_params(axis='both', which='minor', labelsize=18)
    plt.savefig("output/popping_parameter.png", bbox_inches='tight', dpi=500)


def visualize_positional_analysis_popping(fake_renderer):
    """
    Visualize the popping distributions on each grid cell by showing the median time step as background colour and
    by a boxplot
    :param fake_renderer: The fake renderer
    """
    print("visualize_positional_analysis_popping")

    # Plot image with popping distribution
    fig = plt.figure()
    ax = plt.subplot(111)
    img_gray = color.gray2rgb(color.rgb2gray(color.lab2rgb(color.lch2lab(fake_renderer.input_image))))
    ax.imshow(img_gray)
    # Initialize an array for the median popping
    median_pop = np.zeros(fake_renderer.RESOLUTION, np.int32)
    # Loop over all grid cells
    for r in range(GRID_RESOLUTION[0]):
        for c in range(GRID_RESOLUTION[1]):
            median_step = 0
            # If popping artifacts occurred in this block, compute the median
            if fake_renderer.popping_image[r, c]:
                median_step = np.median(fake_renderer.popping_image[r, c])
            # Set all pixels in this block to the median value
            median_pop[
                r * fake_renderer.GRID_CELL_SIZE[0]:(r + 1) * fake_renderer.GRID_CELL_SIZE[0],
                c * fake_renderer.GRID_CELL_SIZE[1]:(c + 1) * fake_renderer.GRID_CELL_SIZE[1]
            ] = median_step
    # Show the median popping array
    im = ax.imshow(
        median_pop,
        vmin=0,
        vmax=fake_renderer.NUM_ITERATIONS,
        cmap=colour_map(fake_renderer, 0.35)
    )
    # Add a color bar
    fig.colorbar(im, label="median step")
    # Remove the frame
    plt.setp(ax, 'frame_on', False)
    # Turn the grid off
    ax.grid('off')
    # Loop over all grid cells
    for r in np.arange(GRID_RESOLUTION[0]):
        for c in np.arange(GRID_RESOLUTION[1]):
            # Show a boxplot for the popping time step distribution inside this grid cell. To make sure the boxplot
            # fits inside the grid cell, we first scale the popping time steps to [0,1]. Then, the origin of the
            # image is 'up', we invert the time steps. Then, we add the grid cell's height to the vertical position
            # of the boxplot. Its horizontal position is at the center of the grid cell, and its width is 75% of the
            # grid cell's width.
            ax.boxplot(
                (1 - 0.025 - 0.95 * (np.array(fake_renderer.popping_image[r, c]) - 2) / (fake_renderer.NUM_ITERATIONS - 2) + r) *
                fake_renderer.GRID_CELL_SIZE[1] - 0.5,
                positions=([(c + 0.5) * fake_renderer.GRID_CELL_SIZE[1] - 0.5]),
                widths=0.75 * fake_renderer.GRID_CELL_SIZE[0])
    # Labels
    ax.set_title(
        "Popping distribution image space\n{:s}".format(title_info(fake_renderer.NUM_NEIGHBOURS, fake_renderer.POWER_PARAMETER)))
    ax.set_xticks([])
    ax.set_yticks([])
    plt.rcParams.update({'font.size': 22})
    plt.savefig(OUTPUT_FOLDER + "popping.png", bbox_inches='tight', dpi=500)


def create_median_pop(fake_renderer, popping_image):
    """
    Visualize the popping distributions on each grid cell by showing the median time step as background colour and
    by a boxplot
    :param fake_renderer:
    :param popping_image: The popping image with a list of popping time steps per grid cell
        """
    # Initialize an array for the median popping
    median_pop = np.zeros(fake_renderer.RESOLUTION, np.int32)
    # Loop over all grid cells
    for cell_r in range(fake_renderer.GRID_RESOLUTION[0]):
        for cell_c in range(fake_renderer.GRID_RESOLUTION[1]):
            median_step = 0
            # If popping artifacts occurred in this block, compute the median
            if popping_image[cell_r, cell_c]:
                median_step = np.median(popping_image[cell_r, cell_c])
            # Set all pixels in this block to the median value
            median_pop[
            cell_r * fake_renderer.GRID_CELL_SIZE[0]:(cell_r + 1) * fake_renderer.GRID_CELL_SIZE[0],
            cell_c * fake_renderer.GRID_CELL_SIZE[1]:(cell_c + 1) * fake_renderer.GRID_CELL_SIZE[1]
            ] = median_step
    return median_pop


def create_popping_image(scheme, popping):
    """

    :param scheme:
    :param popping:
    :return:
    """
    # Initialize an empty list for each parameter combination which will be filled with popping time steps
    popping_image = np.zeros(scheme.GRID_RESOLUTION, dtype=object)  # [(grid position [x,y], step)]
    # Initialize an empty list for each parameter combination which will be filled with popping time steps
    for r in range(scheme.GRID_RESOLUTION[0]):
        for c in range(scheme.GRID_RESOLUTION[1]):
            popping_image[r, c] = []
    # Loop over all popping and append them to the grid cell's lists
    for (r, c), step in popping:
        popping_image[r, c].append(step)
    return popping_image


# ====================================== Visualization (animation) ===============================================

def animate_reconstruction(fake_renderer):
    """
    Makes an animation of the reconstructed images.
    """
    print("animate_reconstruction")

    # Initialization function for animation
    def anim_init():
        return [im]

    # Animation function
    def anim_func(i):
        # Get the reconstructed image from the stack
        im.set_array(fake_renderer.reconstructions[i])
        # Update title
        plt.title("R_n: {:d}/{:d}\n{:s}".format(fake_renderer.FRAMES[i] + 1,
                                                fake_renderer.NUM_ITERATIONS,
                                                title_info(fake_renderer.NUM_NEIGHBOURS, fake_renderer.POWER_PARAMETER)))
        # Return the frame
        return [im]

    # Setup figure
    fig = plt.figure()
    plt.rcParams.update({'font.size': 10})
    plt.axis('off')
    im = plt.imshow(np.zeros(fake_renderer.RESOLUTION, dtype="float64"), interpolation='none')

    # Create the animation
    anim = animation.FuncAnimation(
        fig,
        func=anim_func,
        init_func=anim_init,
        frames=fake_renderer.NUM_FRAMES,
        blit=True
    )
    # Save the animation
    anim.save(
        filename=OUTPUT_FOLDER + "animate_reconstruction.gif",
        dpi=300,
        fps=FPS
    )


def animate_refinement_characteristics_similarity(fake_renderer):
    """
    Makes an animation of the similarity graph.
    :param fake_renderer: The fake renderer
    :return:
    """
    print("animate_refinement_characteristics_similarity")

    # Setup figure
    fig = plt.figure()
    ax = plt.axes(
        xlim=[1.5, fake_renderer.NUM_ITERATIONS + 0.5],
        ylim=[0.95 * np.min(fake_renderer.similarity), 1 * 1.05]
    )
    line, = ax.plot([], [], lw=2)

    # lists to store x and y axis points
    xdata, ydata = [], []

    # initialization function: plot the background of each frame
    def anim_init():
        line.set_data([], [])
        return line,

    # animation function.  This is called sequentially
    def anim_func(i):
        xdata.append(i + 2)
        ydata.append(fake_renderer.similarity[i])
        line.set_data(xdata, ydata)
        return line,

    # Set labels
    ax.set_title("Reconstructed image similarity\n{:s}".format(title_info(fake_renderer.NUM_NEIGHBOURS, fake_renderer.POWER_PARAMETER)))
    ax.set_ylabel("SSIM")
    ax.set_xlabel("step")

    # Create the animation
    anim = animation.FuncAnimation(
        fig,
        func=anim_func,
        init_func=anim_init,
        frames=fake_renderer.NUM_ITERATIONS - 1,
        blit=True
    )
    # Save the animation
    anim.save(
        filename=OUTPUT_FOLDER + "animate_similarity.gif",
        dpi=300,
        fps=FPS
    )


def animate_positional_analysis_popping(fake_renderer):
    """
    Makes an animation of the popping image.
    :param fake_renderer: The fake renderer
    :return:
    """
    print("animate_positional_analysis_popping")

    # Get the specified colour map
    cmap = plt.get_cmap(COLOUR_MAP, 2 * fake_renderer.NUM_ITERATIONS)
    # Extract the colours
    cmap_colours = cmap(np.linspace(0, 1, 2 * fake_renderer.NUM_ITERATIONS))
    # Add background colour to the list of colours
    cmap_colours = np.vstack((BACKGROUND_COLOUR, cmap_colours[::-1]))
    # Set transparency of the colours
    cmap_colours[1:, 3] = 0.35
    # Crate the new colour map
    cmap = ListedColormap(cmap_colours)

    # Setup the figure
    fig, ax = plt.subplots()
    # Compute a gray-scale version of the input image
    img_gray = color.gray2rgb(color.rgb2gray(color.lab2rgb(color.lch2lab(fake_renderer.input_image))))

    def anim_init():
        return [ax.imshow(img_gray, interpolation='none')]

    def anim_func(i):
        # Clear the figures
        ax.clear()
        ax.axis('off')
        # Plot the image in grayscale (background)
        ax.imshow(img_gray)

        popping_image = fake_renderer.popping_image.copy()
        for r in range(fake_renderer.GRID_RESOLUTION[0]):
            for c in range(fake_renderer.GRID_RESOLUTION[1]):
                popping_image[r, c] = [step for step in popping_image[r, c] if step <= i+1]

        # Show the median popping array
        im = ax.imshow(create_median_pop(fake_renderer, popping_image), vmin=0, vmax=fake_renderer.NUM_ITERATIONS,
                       cmap=cmap)
        # Turn the grid off
        plt.grid('off')
        # Loop over all grid cells
        for cell_r in np.arange(GRID_RESOLUTION[0]):
            for cell_c in np.arange(GRID_RESOLUTION[1]):
                # Show a boxplot for the popping time step distribution inside this grid cell. To make sure the boxplot
                # fits inside the grid cell, we first scale the popping time steps to [0,1]. Then, the origin of the
                # image is 'up', we invert the time steps. Then, we add the grid cell's height to the vertical position
                # of the boxplot. Its horizontal position is at the center of the grid cell, and its width is 75% of the
                # grid cell's width.
                ax.boxplot(
                    (1 - 0.025 - 0.95 * (np.array(popping_image[cell_r, cell_c]) - 2) / (fake_renderer.NUM_ITERATIONS - 2) + cell_r) *
                    fake_renderer.GRID_CELL_SIZE[1] - 0.5,
                    positions=([(cell_c + 0.5) * fake_renderer.GRID_CELL_SIZE[1] - 0.5]),
                    widths=0.75 * fake_renderer.GRID_CELL_SIZE[0])
        plt.title("P_n: {:d}/{:d}\n{:s}".format(fake_renderer.FRAMES[i] + 1,
                                                fake_renderer.NUM_ITERATIONS,
                                                title_info(fake_renderer.NUM_NEIGHBOURS, fake_renderer.POWER_PARAMETER)))
        return [im]

    anim = animation.FuncAnimation(
        fig,
        func=anim_func,
        init_func=anim_init,
        frames=fake_renderer.NUM_FRAMES,
        blit=True
    )
    anim.save(
        filename=OUTPUT_FOLDER + "animate_popping.gif",
        dpi=300,
        fps=FPS
    )


def animate_positional_analysis_sampling_time(fake_renderer):
    """
    Makes an animation of when the pixels were sampled. Moreover, this is overlaid on top of a gray-scale version of
    the reconstructed images, so that we can better understand why those pixels were sampled
    """
    print("animate_positional_analysis_sampling_time")

    # Initialization function for animation
    def anim_init():
        return [ax.imshow(np.zeros(fake_renderer.RESOLUTION, dtype=np.float64))]

    # Animation function
    def anim_func(i):
        # Clear the figures
        ax.clear()
        ax.axis('off')
        # Plot the image in grayscale (background)
        ax.imshow(color.gray2rgb(color.rgb2gray(fake_renderer.reconstructions[i])), interpolation='none')
        # Make a copy of scheme.when
        when_copy = fake_renderer.when.copy()
        # We only want to display pixels up to and including step scheme.frames[i]
        for (r, c) in np.argwhere(fake_renderer.when > fake_renderer.FRAMES[i] + 1):
            when_copy[r, c] = 0
        # Plot the updated when-array
        im = ax.imshow(
            when_copy,
            vmin=0,
            vmax=fake_renderer.NUM_ITERATIONS,
            cmap=cmap,
            interpolation='none')
        # Update title
        plt.title("W_n: {:d}/{:d}\n{:s}".format(fake_renderer.FRAMES[i] + 1,
                                                fake_renderer.NUM_ITERATIONS,
                                                title_info(fake_renderer.NUM_NEIGHBOURS, fake_renderer.POWER_PARAMETER)))
        # plt.rcParams.update({'font.size': 14})
        # Return the frame
        return [im]

    # Setup figure
    fig, ax = plt.subplots()
    plt.rcParams.update({'font.size': 10})

    # Create a colour map with 65% transparency
    cmap = colour_map(fake_renderer, 0.65)

    # Create the animation
    anim = animation.FuncAnimation(
        fig,
        func=anim_func,
        init_func=anim_init,
        frames=fake_renderer.NUM_FRAMES,
        blit=True
    )
    # Save the animation
    anim.save(
        filename=OUTPUT_FOLDER + "animate_when.gif",
        dpi=300,
        fps=FPS
    )
