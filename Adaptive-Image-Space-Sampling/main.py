# Scientific
import numpy as np
# Graphics
from skimage import io, color
# Misc
import time
import multiprocessing as mp
# Project
from FakeRenderer import FakeRenderer
from settings import *
from visualize import visualize_results


def parallel_rendering_process(fr):
    """
    Run the rendering process of a fake renderer in a separate function (used for multiprocessing)
    :param fr: Fake renderer
    :return:
    """
    fr.rendering_process()
    return fr


if __name__ == '__main__':
    # Start timer
    print("=== Rendering processes ===")
    start = time.time()

    # Load the image
    img = io.imread(FILE)
    # Remove the alpha channel
    img = img[:, :, :3]
    # Scale to [0,1]
    img = img / 255

    # A list store multiple fake renderers
    fake_renderers = []

    # Create a fake renderer for each parameter combination
    for n in NUM_NEAREST_NEIGHBOURS:
        for p in POWER_PARAMETERS:
            for _ in range(NUM_RUNS):
                # Initialize the scheme
                fake_renderer = FakeRenderer(
                    img,
                    sample_percentage=PERCENTAGE_PPI,
                    num_neighbours=n,
                    power_parameter=p,
                    pop_threshold=POP_THRESHOLD,
                    pop_group_size=POP_GROUP_SIZE,
                    grid_resolution=GRID_RESOLUTION
                )

                # Add fake renderer to the list
                fake_renderers.append(fake_renderer)

    # Do the rendering process in parallel
    p = mp.Pool(NUM_CPUS)
    frs = p.map(parallel_rendering_process, fake_renderers)
    p.close()
    p.join()

    # Visualize the results
    visualize_results(frs)
