# Misc
import random
import multiprocessing as mp
import re
import time
# File
import zipfile
import bz2
import pandas as pd
# Project
from settings import *
from visualize import visualize_results


# ====================================== Helper functions ===============================================

# Decompose a number into its 4^y factors: num = x0*4^(h-1) + x1*4^(h-2) + ... + x(h-1)*4^0
def fac4(num, h):
    arr = np.zeros((h,), np.int32)
    fit = 0
    for x in range(h):
        arr[x] = (num - fit) // 4 ** (h - x - 1)
        fit += arr[x] * 4 ** (h - x - 1)
    return arr


# Create a lookup table for each number' fac4
def init_fac4_lookup(num, h):
    print("start init_fac4_lookup")
    fac4_lookup = np.empty((num, h), dtype=np.int32)
    for x in range(num):
        fac4_lookup[x] = fac4(x, h)
    print("end init_fac4_lookup")
    return fac4_lookup


# Create a lookup table for the distances between images
def init_position_lookup(num, h, fac4_lookup):
    # The positions of the images in the grid
    index2pos = np.zeros((num, 2), dtype=np.int32)
    pos2index = np.zeros((GRID_DIM, GRID_DIM), dtype=np.int32)
    # Position in a quad
    quad_pos = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    # Fill array
    for x in range(num):
        x_fac4 = fac4_lookup[x, :]
        for h_ in range(h):
            index2pos[x] += quad_pos[x_fac4[h_]] * 2 ** (h - h_ - 1)
        pos = index2pos[x]
        pos2index[pos[0], pos[1]] = x
    return index2pos, pos2index


# Reshape the grid layout from 1D to 2D
def reshape_grid(grid_1D, index2pos):
    grid_2D = np.zeros((GRID_DIM, GRID_DIM), dtype=np.int32) - 1
    for idx, x in enumerate(grid_1D):
        r, c = index2pos[idx]
        grid_2D[r, c] = x
    return grid_2D


# ====================================== Process data ===============================================

def get_data(index2pos):
    """
    Read the costs from the csv log files
    :return: A 2D array of the costs
    """

    def get_costs():
        """
        Get the costs for each iteration of each run
        :return:
        """
        # Initialize cost array
        costs = np.zeros((NUM_RUNS, NUM_TOTAL_ITERATIONS + 1), dtype=np.float64)

        # Open the zip-file
        with zipfile.ZipFile(ZIP_FILE, "r") as f:
            # Store all the file names we want to read (in this case all the csv files)
            file_names = []
            for name in f.namelist():
                if "MAC" not in name and name.endswith(".csv"):
                    file_names.append(name)

            # Sort the files
            file_names.sort()
            # Keep the first the iterations for NUM_RUNS runs
            file_names = file_names[:NUM_RUNS * NUM_SUPER_ITERATIONS]

            # Loop through the files and get the costs
            file_num = 0
            for name in file_names:
                # Extract the super iteration and run identifier
                x = re.findall("\d\d\d\d+", name)
                # Convert string to int
                run = int(x[1])
                super_iteration = int(x[0]) - 1

                print("get_cost - File:", file_num, " name:", name, " run:", run, " super-iteration:", super_iteration)

                # Read the csv (only the third column which contains the costs)
                df = pd.read_csv(f.open(name), usecols=[2], header=None)
                # Convert to numpy array and ravel
                costs_arr = df.to_numpy().ravel()
                # Store the costs
                if super_iteration == 0:
                    # For the first super-iteration store all NUM_ITERATIONS+1 costs
                    costs[run, :NUM_SUB_ITERATIONS + 1] = costs_arr
                else:
                    # For the other super-iterations skip the first one (as we already have it from the previous
                    # super-iterations)
                    costs[run, (super_iteration * NUM_SUB_ITERATIONS + 1):(
                            (super_iteration + 1) * NUM_SUB_ITERATIONS + 1)] = costs_arr[1:]

                file_num += 1

            # Close the zip-file
            f.close()

        # Return the costs
        return costs

    def determine_popping_artifacts():
        """
        Determine when popping artifacts occurred and create small images showing where they happened
        :return:
        """
        # Array with popping for each run
        popping = np.abs(np.diff(costs, axis=1)) >= POP_THRESHOLD
        # Array to store some small images for each run. Initial value is ITERATION_BLOCK + 1, which will be used to
        # indicate no popping happened inside this iteration block
        small_images = np.zeros((NUM_RUNS, NUM_ITERATION_BLOCKS, GRID_DIM, GRID_DIM)) + ITERATION_BLOCK_SIZE + 1

        # Open the fip-file
        with zipfile.ZipFile(ZIP_FILE, "r") as f:
            # Store all the file names we want to read (in this case all the csv files)
            file_names = []
            for name in f.namelist():
                if "MAC" not in name and "qt" not in name and name.endswith(".bz2"):
                    file_names.append(name)

            # Sort the files
            file_names.sort()
            # Keep the files of the first NUM_SUPER_ITERATIONS super-iterations
            file_names = file_names[:NUM_RUNS * NUM_SUPER_ITERATIONS * NUM_SUB_ITERATIONS]

            # For storing the previous grid layout
            prev_config = np.zeros(NUM_IMAGES, dtype=np.int32)

            # Loop through the files and get the costs
            file_num = 0
            for name in file_names:
                # Extract the super iteration and run identifier
                x = re.findall("\d\d\d\d+", name)
                # Convert string to int
                run = int(x[1])
                super_iteration = int(x[0]) - 1
                sub_iteration = int(x[3])

                # Read the data from the bz2 file
                data_buf = bz2.BZ2File(f.open(name)).read()
                # Convert data to integers
                config = np.frombuffer(data_buf, dtype=np.int32)

                # Compute the actual iteration number
                iteration = super_iteration * NUM_SUB_ITERATIONS + sub_iteration

                # From the second iteration, compute the differences with the previous layouts, if a popping happened
                # at this iteration
                if iteration > 0 and popping[run, iteration]:
                    # The grid block in the visualization
                    iteration_block = iteration // ITERATION_BLOCK_SIZE
                    # Set the initial values to zero
                    if small_images[run, iteration_block, 0, 0] == ITERATION_BLOCK_SIZE + 1:
                        small_images[run, iteration_block] = 0
                    # Add the tile differences to the grid block
                    small_images[run, iteration_block] += (
                            reshape_grid(prev_config, index2pos) != reshape_grid(config, index2pos)).astype(
                        np.float64)

                # Store the grid layout
                prev_config = config.copy()

                print("compute_pop - File:", file_num, " name:", name, " run:", run)

                file_num += 1

            # Close zip-file
            f.close()

        # Return the popping and small images
        return popping, small_images

    # Call functions
    costs = get_costs()
    popping, small_imgs = determine_popping_artifacts()
    return costs, popping, small_imgs


def get_data_parameter_study():
    """
    Get the costs for each iteration of each run
    :return:
    """

    NUM_RUNS_P = 16
    NUM_PARAMETERS_P = 5
    NUM_TOTAL_RUNS_P = NUM_RUNS_P * NUM_PARAMETERS_P

    # Initialize cost array
    costs = np.zeros((NUM_TOTAL_RUNS_P, NUM_TOTAL_ITERATIONS + 1), dtype=np.float64)
    computation_time = np.zeros((NUM_TOTAL_RUNS_P, NUM_TOTAL_ITERATIONS + 1), dtype=np.float64)

    # Open the zip-file
    with zipfile.ZipFile(ZIP_FILE_PARAMETERS, "r") as f:
        # Store all the file names we want to read (in this case all the csv files)
        file_names = []
        for name in f.namelist():
            if "MAC" not in name and name.endswith(".csv"):
                file_names.append(name)

        # Sort the files
        file_names.sort()

        # Loop through the files and get the costs
        file_num = 0
        for name in file_names:
            # Extract the super iteration and run identifier
            x = re.findall("\d\d\d\d+", name)
            # Convert string to int
            run = int(x[1])
            super_iteration = int(x[0]) - 1

            # Extract parameter value
            x = re.findall("\d+", name)
            k = int(x[0])

            y = -1
            if k == 2:
                y = 0
            elif k == 5:
                y = 1
            elif k == 20:
                y = 3
            elif k == 40:
                y = 4

            print("get_cost - File:", file_num, " name:", name, " k:", k, " run:", run, " super-iteration:",
                  super_iteration)

            # Read the csv (only the third column which contains the costs)
            df = pd.read_csv(f.open(name), usecols=[2], header=None)
            # Convert to numpy array and ravel
            costs_arr = df.to_numpy().ravel()
            # Store the costs
            costs[run + y * NUM_RUNS_P, :NUM_SUB_ITERATIONS + 1] = costs_arr

            # Read the csv (only the third column which contains the costs)
            df = pd.read_csv(f.open(name), usecols=[1], header=None)
            # Convert to numpy array and ravel
            time_arr = df.to_numpy().ravel()
            # Store the costs
            computation_time[run + y * NUM_RUNS_P, :NUM_SUB_ITERATIONS + 1] = time_arr

            file_num += 1

        # Close the zip-file
        f.close()

        # Open the zip-file
        with zipfile.ZipFile(ZIP_FILE, "r") as f:
            # Store all the file names we want to read (in this case all the csv files)
            file_names = []
            for name in f.namelist():
                if "MAC" not in name and name.endswith(".csv"):
                    file_names.append(name)

            # Sort the files
            file_names.sort()

            # Loop through the files and get the costs
            file_num = 0
            for name in file_names:
                # Extract the super iteration and run identifier
                x = re.findall("\d\d\d\d+", name)
                # Convert string to int
                run = int(x[1])
                super_iteration = int(x[0]) - 1

                # Extract parameter value
                x = re.findall("\d+", name)
                k = int(x[0])

                y = 2

                print("get_cost - File:", file_num, " name:", name, " k:", k, " run:", run, " super-iteration:",
                      super_iteration)

                # Read the csv (only the third column which contains the costs)
                df = pd.read_csv(f.open(name), usecols=[2], header=None)
                # Convert to numpy array and ravel
                costs_arr = df.to_numpy().ravel()
                # Store the costs
                costs[run + y * NUM_RUNS_P, :NUM_SUB_ITERATIONS + 1] = costs_arr

                # Read the csv (only the third column which contains the costs)
                df = pd.read_csv(f.open(name), usecols=[1], header=None)
                # Convert to numpy array and ravel
                time_arr = df.to_numpy().ravel()
                # Store the costs
                computation_time[run + y * NUM_RUNS_P, :NUM_SUB_ITERATIONS + 1] = time_arr

                file_num += 1

            # Close the zip-file
            f.close()

    # Array with popping for each run
    NUM_T = 3
    popping_all_1 = np.abs(np.diff(costs, axis=1)) >= 0.005
    popping_all_2 = np.abs(np.diff(costs, axis=1)) >= 0.01
    popping_all_3 = np.abs(np.diff(costs, axis=1)) >= 0.02
    popping = np.zeros((NUM_PARAMETERS_P, NUM_T), dtype=object)
    for k in range(NUM_PARAMETERS_P):
        for t in range(NUM_T):
            popping[k, t] = []
    for k in range(NUM_PARAMETERS_P):
        for i in range(NUM_RUNS_P):
            for t in range(NUM_T):
                when = np.array([-100])
                if t == 0:
                    when = np.where(popping_all_1[k * NUM_RUNS_P + i])[0] + 1
                elif t == 1:
                    when = np.where(popping_all_2[k * NUM_RUNS_P + i])[0] + 1
                elif t == 2:
                    when = np.where(popping_all_3[k * NUM_RUNS_P + i])[0] + 1
                popping[k, t].append(when.tolist())

    # Return the costs
    return costs, popping, computation_time


if __name__ == "__main__":
    # Initialize arrays
    fac4_lookup = init_fac4_lookup(NUM_IMAGES, h)
    index2pos, pos2idx = init_position_lookup(NUM_IMAGES, h, fac4_lookup)

    # Process data
    costs, popping, small_images = get_data(index2pos)
    data_normal = {
        "costs": costs,
        "popping": popping,
        "small_images": small_images
    }
    costs, popping, computation_time = get_data_parameter_study()
    data_parameter_study = {
        "costs": costs,
        "popping": popping,
        "computation_time": computation_time
    }

    # Visualize results
    visualize_results(data_normal, data_parameter_study)
