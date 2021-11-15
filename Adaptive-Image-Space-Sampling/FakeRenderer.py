# Scientific
import numpy as np
from scipy.spatial import distance
from skimage.metrics import structural_similarity as ssim
from skimage import color
# Misc
import time
import random
import sys
# Project
from settings import *


class FakeRenderer:
    """
    This is an iterative refinement process of a “fake” renderer. Instead of actually obtaining
    real pixel values by sending a ray, this fake renderer just draws values
    from an image we already have. The sampling is based on nearest neighbours
    and how much they differ. The image is reconstructed using IDW. Image
    similarity is measured using the structural similarity index.
    """
    def __init__(self,
                 img,
                 num_iterations=-1,
                 sample_percentage=0.1,
                 num_neighbours=4,
                 power_parameter=4,
                 pop_threshold=40,
                 pop_group_size=8,
                 grid_resolution=(16, 16)
                 ):
        # Get image sizes
        rows = img.shape[0]
        cols = img.shape[1]

        # Compute constants
        #
        # The image resolution
        self.RESOLUTION = (rows, cols)
        # The total number of pixels
        self.TOTAL_PIXELS = rows * cols
        # Number of Pixels Per Iteration
        self.SAMPLE_SIZE = np.ceil(sample_percentage * self.TOTAL_PIXELS).astype(int)
        # Decide on the number of iterations (either user specified or such that
        # all pixels will be sampled)
        # Number of iterations required to sample all pixels
        NUM_ITERATIONS_FOR_ALL = np.ceil(self.TOTAL_PIXELS / self.SAMPLE_SIZE).astype(int)
        if num_iterations == -1:
            # Default case covers all pixels
            self.NUM_ITERATIONS = NUM_ITERATIONS_FOR_ALL
        else:
            # User specified (the process does not need to take longer than
            # NUM_ITERATIONS_FOR_ALL iterations)
            self.NUM_ITERATIONS = np.min([num_iterations, NUM_ITERATIONS_FOR_ALL])
        # Number of neighbours to consider for sampling
        self.NUM_NEIGHBOURS = num_neighbours
        # Thw power parameter for IDW
        self.POWER_PARAMETER = power_parameter
        # The minimum colour distance for considering popping artifacts
        self.POP_THRESHOLD = pop_threshold
        # The minimum number of pixels to pop for capturing them
        self.POP_GROUP_SIZE = pop_group_size
        # The grid resolution, i.e. the number of grid cells in each dimension (for capturing the popping)
        self.GRID_RESOLUTION = grid_resolution
        # The size of the grid cells
        self.GRID_CELL_SIZE = (
            np.ceil(self.RESOLUTION[0] / self.GRID_RESOLUTION[0]).astype(int),
            np.ceil(self.RESOLUTION[1] / self.GRID_RESOLUTION[1]).astype(int)
        )

        # Initialize variables for recording (i.e. making animation)
        #
        # Number of frames
        self.NUM_FRAMES = 100
        # Frame numbers
        self.FRAMES = np.unique(np.linspace(0, self.NUM_ITERATIONS - 1, self.NUM_FRAMES, dtype=np.int32))
        # Make sure that no frames are duplicates
        self.NUM_FRAMES = min(self.NUM_FRAMES, len(self.FRAMES))

        # Initialize variables for the process
        #
        # The 'final image' which is we already have in this toy example
        self.input_image = color.lab2lch(color.rgb2lab(img))
        # The final image, the render so to say (iteratively more values will be obtained)
        self.render = np.empty(self.RESOLUTION + (3,), dtype=np.float64)
        # The number of available pixels
        self.num_available_pixels = self.TOTAL_PIXELS
        # Norm for reconstruction
        self.norm = np.zeros(self.RESOLUTION, dtype=np.float64)
        # The nearest neighbours (that is, nearest sampled pixels) of the pixels (initialized to -1)
        self.nearest_neighbours = np.zeros(self.RESOLUTION + (self.NUM_NEIGHBOURS, 2), dtype=np.int32) - 1
        # The reconstructed image
        self.reconstructed_image = np.zeros(self.RESOLUTION + (3,), dtype=np.float64)

        # Initialize variables for the visualization
        #
        # A stack of (reconstructed) images (one for every frame)
        self.reconstructions = np.empty((self.NUM_FRAMES,) + self.RESOLUTION + (3,), dtype=np.float64)
        # Stores when pixels were sampled
        self.when = np.zeros(self.RESOLUTION, dtype=np.int32)
        # Stores the similarity metric
        self.similarity = np.zeros(self.NUM_ITERATIONS - 1, dtype=np.float64)
        # Stores the similarity metric
        self.similarity_final = np.zeros(self.NUM_ITERATIONS - 1, dtype=np.float64)
        # Stores the colour distance for each pixel
        self.signal = np.zeros(self.RESOLUTION + (self.NUM_ITERATIONS,), dtype=np.float64)
        # Initialize an empty list for each block which will be filled with popping time steps
        self.popping_image = np.zeros(self.GRID_RESOLUTION, dtype=object)  # [(grid position [x,y], step)]
        for r in range(self.GRID_RESOLUTION[0]):
            for c in range(self.GRID_RESOLUTION[1]):
                self.popping_image[r, c] = []
        # Stores the previously sampled pixels
        self.prev_sample = np.array([])
        # Binary vector that stores at which time step popping artifacts occurred
        self.popping_arr = np.zeros((self.NUM_ITERATIONS - 1), dtype=bool)
        # Compute the number of intervals
        self.NUM_INTERVALS = np.ceil(self.NUM_ITERATIONS / INTERVAL_SIZE).astype(np.int32)
        # Array to store some small images for each run. Initial value is ITERATION_BLOCK + 1, which will be used to
        # indicate no popping happened inside this iteration block
        self.small_images = np.zeros((self.NUM_INTERVALS,) + self.RESOLUTION) + INTERVAL_SIZE + 1

    def sample_metric(self, u):
        """
        Compute the metric of unsampled pixel u by computing the colour distance (in CIE LAB) between its nearest
        neighbours.
        :param u: The pixel candidate for which we compute the sampling metric
        :return: The metric
        """
        r, c = u
        if self.nearest_neighbours[r, c, 0, 0] == -1:
            # After the initial random sample, the nearest neighbours are those samples
            comb = self.prev_sample
        else:
            # Concatenate the pixel's nearest neighbours and the previous sample
            comb = np.concatenate((self.nearest_neighbours[r, c], self.prev_sample), axis=0)
        # Compute the distances
        dist = distance.cdist(np.array([u]), comb)
        # Get the indices of the nearest pixels
        idx = (dist.ravel()).argsort()[:self.NUM_NEIGHBOURS]
        # Extract them from comb
        nearest_neighbours = comb[idx, :]
        # Store the nearest neighbours for this pixel
        self.nearest_neighbours[r, c] = nearest_neighbours
        # Get the colours of the nearest neighbours
        colour_arr = self.input_image[nearest_neighbours[:, 0], nearest_neighbours[:, 1]]

        # Compute the metric: the sum of the distances between the colours
        m = np.sum(distance.pdist(colour_arr))
        # If it is zero (e.g. in constant regions), return the smallest float number (so that np.random.choice
        # does not complain about not having enough non-zero values
        if m == 0.0:
            return sys.float_info.min
        return m

    def sample_lottery(self, step):
        """
        Sample new pixels based on sample set S. New pixels are chosen by looking at the image variation in
        self.NUM_NEIGHBOURS closest samples and using that metric as weights for weighted random sampling
        :param step: The time step
        :return: The new sample
        """
        if step == 1:
            # Initial sample is random
            sample = np.c_[np.unravel_index(random.sample(range(self.TOTAL_PIXELS), self.SAMPLE_SIZE), self.RESOLUTION)]
        else:
            # Get the set of unsampled pixels U
            U = np.argwhere(self.when == 0)

            # Initialize array to store some metric (i.e. the colour variation)
            metric = np.empty(U.shape[0])
            # Loop through each unsampled pixel
            for idx, u in enumerate(U):
                # Compute and store the metric
                metric[idx] = self.sample_metric(u)

            # Convert metrics to weights in [0,1]
            metric = metric / np.sum(metric)
            # Sample size cannot be large then number of available pixels
            sample_size = np.min([self.num_available_pixels, self.SAMPLE_SIZE])
            # Check if we are at the final set of pixels
            if sample_size == self.num_available_pixels:
                # If so, the sample is all pixels in U
                sample = U
            else:
                # Randomly select the indices of sample_size pixels, where metric are the weights
                sample_idx = np.random.choice(U.shape[0], size=sample_size, replace=False, p=metric)
                # Get the pixel locations
                sample = U[sample_idx, :]

        # Sample bookkeeping
        self.sample_bookkeeping(sample, step)

        # Obtain sample values
        self.obtain_sample_values(sample)

        # Store the sample
        self.prev_sample = sample.copy()

        # Return the sample
        return sample

    def sample_bookkeeping(self, sample, step):
        """
        Records how many and which pixels were sampled at this step
        :param sample: The sampled pixels
        :param step: The time step
        :return:
        """
        # Sampled pixels are now unavailable
        self.num_available_pixels -= sample.shape[0]

        # Store when these pixels were sampled
        for (r, c) in sample:
            # In this implementation pixels cannot be sampled more than once
            if self.when[r, c] != 0:
                print("ERROR: ({:d},{:d}) was already sampled at step {:d}, but again at {:d}!".format(
                    r, c,
                    int(self.when[r, c]),
                    step
                ))
            # This pixel (r,c) changed at this step
            self.when[r, c] = step

    def obtain_sample_values(self, sample):
        """
        Obtain sample values. In this "fake" renderer we already know the
        samples values. In practical applications, here we would shoot a ray for
        the newly sampled pixels
        :param sample: The sampled pixels
        :return:
        """
        # Obtain the value for this pixel
        for (r, c) in sample:
            self.render[r, c] = self.input_image[r, c]

    def reconstruct_single(self, u):
        """
        Reconstruct a single pixel using IDW
        :param u: The unsampled pixel
        :return:
        """
        r, c = u
        # Array of simple IDW weighting function
        w_i = 1 / distance.cdist(np.array([u]), self.prev_sample) ** self.POWER_PARAMETER
        # Get a list of colours from the previously sampled pixels
        V = self.input_image[self.prev_sample[:, 0], self.prev_sample[:, 1]]
        # Get the old norm
        old_norm = self.norm[r, c]
        # Compute the new norm
        self.norm[r, c] = self.norm[r, c] + np.sum(w_i)
        # Compute interpolated value
        return (self.reconstructed_image[r, c] * old_norm + np.sum(V * w_i.T, 0)) / self.norm[r, c]

    def reconstruct(self):
        """
        Reconstruct an image using S with Inverse Distance Weighting
        :return:
        """
        # Get the set of sampled (S) and unsampled (U) pixels
        S = np.argwhere(self.when != 0)
        U = np.argwhere(self.when == 0)

        # Set the colours at the samples (known values)
        for (r, c) in S:
            self.reconstructed_image[r, c] = self.render[r, c]

        # Compute the colours at the unsampled pixels (unknown values)
        for (r, c) in U:
            self.reconstructed_image[r, c] = self.reconstruct_single((r, c))

        # Return the refined image
        return self.reconstructed_image

    def compute_visualizations(self, step, R, R_prev):
        """
        Detect any popping artifacts
        :param step: The time step
        :param R: The most recent reconstructed image
        :param R_prev: The previously reconstructed image
        :return:
        """
        # Convert the reconstructed LCH images to the CIE LAB colour space and compute the Euclidean distance between
        c_dist = color.deltaE_cie76(color.lch2lab(R), color.lch2lab(R_prev))
        # Store the distances for the signal plot
        self.signal[:, :, step-2] = c_dist.copy()
        # Compute the similarity metric between reconstructed images (e.g. the structural similarity index)
        self.similarity[step - 2] = ssim(R, R_prev, multichannel=True)
        # Compute the similarity metric between the reconstructed image and the final image (e.g. the structural
        # similarity index)
        self.similarity_final[step - 2] = ssim(R, self.input_image, multichannel=True)

        # Create the popping image
        popping = c_dist >= self.POP_THRESHOLD

        if np.any(popping):
            # The grid block in the visualization
            iteration_block = (step-2) // INTERVAL_SIZE
            # Set the initial values to zero
            if self.small_images[iteration_block, 0, 0] == INTERVAL_SIZE + 1:
                self.small_images[iteration_block] = 0
            # Add the tile differences to the grid block
            self.small_images[iteration_block] += popping

        # For detecting the popping, we expand the grid cells by 50% in each direction
        expansion = self.GRID_CELL_SIZE[0] // 2, self.GRID_CELL_SIZE[1] // 2
        # Loop over all grid cells
        for cell_r in range(self.GRID_RESOLUTION[0]):
            for cell_c in range(self.GRID_RESOLUTION[1]):
                # Compute the positions (make sure its within the image boundaries)
                up = max(0, cell_r * self.GRID_CELL_SIZE[0] - expansion[0])
                down = min(self.RESOLUTION[0], (cell_r + 1) * self.GRID_CELL_SIZE[0] + expansion[0])
                left = max(0, cell_c * self.GRID_CELL_SIZE[1] - expansion[1])
                right = min(self.RESOLUTION[1], (cell_c + 1) * self.GRID_CELL_SIZE[1] + expansion[1])
                # Check how many pixels are above POP_THRESHOLD and if there are at least POP_GROUP_SIZE
                if np.sum(popping[up:down, left:right]) >= self.POP_GROUP_SIZE:
                    # All popping
                    # self.popping_at.append(((cell_r, cell_c), step))
                    self.popping_image[cell_r, cell_c].append(step)
                    self.popping_arr[step-2] = True

    def rendering_process(self):
        """
        The rendering process consists of iteratively sampling and
        reconstructing (i.e. iterative refinement)
        :return:
        """
        # Print starting message and table header
        header = " {:{width}s} | {:{width}s} | {:{width}s} | {:{width}s} | {:{width}s} | {:{width}s}".format(
            "step",
            "total rays",
            "add. rays",
            "%",
            "total time (s)",
            "time int. (s)",
            width=14
        )
        print("== Start ==", header, "-" * len(header), sep='\n')

        # Keep track of total elapsed time
        total_time = 0

        # Array to store the previous reconstructed image
        R_prev = np.zeros(self.RESOLUTION + (3,), np.float64)

        # Keep track of the frames
        frame = 0

        # Loop (iterative refinement)
        for step in np.arange(self.NUM_ITERATIONS) + 1:
            # start timer
            start = time.time()

            ####################################################################
            # Main part of the process
            ####################################################################
            # Get new samples
            sample = self.sample_lottery(step)
            # Reconstruct the image
            R = self.reconstruct()
            # Only compute the visualizations for step > 1
            if step > 1:
                # Compute metrics
                self.compute_visualizations(step, R, R_prev)
            ####################################################################

            # If we are recording and the current step is one of the frames, store the reconstructed image
            if step - 1 == self.FRAMES[frame]:
                self.reconstructions[frame] = color.lab2rgb(color.lch2lab(R))
                frame += 1

            # Store R_prev
            R_prev = R.copy()

            # stop timer
            end = time.time()
            # Add elapsed time
            total_time += end - start

            # Print progress
            print("{:{width}d}  | {:{width}d} | {:{width}d} | {:{width}.2f} | {:{width}.2f} | {:{width}.3f}".format(
                step,  # step
                self.TOTAL_PIXELS - self.num_available_pixels,  # total rays
                sample.shape[0],  # Add. rays
                100 * (self.TOTAL_PIXELS - self.num_available_pixels) / self.TOTAL_PIXELS,  # percentage
                total_time,  # total time
                end - start,  # time interval
                width=14,
            ))

        # Create a circle to enlarge the pixel by taking the maximum value in the circle radius (pixels in higher
        # resolution images are barely visible, so this improves visual)
        radius = 5
        kernel = np.zeros((2*radius + 1, 2*radius+1), dtype=bool)
        for ii in range(2*radius+1):
            for jj in range(2*radius+1):
                kernel[ii, jj] = (ii-radius)**2 + (jj-radius)**2 <= radius**2
        # Pad the small_image
        sm_padded = np.zeros((self.NUM_INTERVALS,) + (GRID_DIM+2*radius, GRID_DIM+2*radius))
        for sm in range(self.NUM_INTERVALS):
            sm_padded[sm, radius:(GRID_DIM+radius), radius:(GRID_DIM+radius)] = self.small_images[sm].copy()
            for b_r in range(self.RESOLUTION[0]):
                for b_c in range(self.RESOLUTION[1]):
                    sub = sm_padded[sm, (b_r):(b_r+2*radius+1), (b_c):(b_c+2*radius+1)]
                    # Take the maximum value
                    self.small_images[sm, b_r, b_c] = np.max(sub[kernel])

        # Print ending message
        print("== Done ==")
