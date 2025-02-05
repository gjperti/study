import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from matplotlib.colors import ListedColormap


# Building the grid

def build_empty_grid(grid_size):
    grid = np.zeros((grid_size, grid_size))
    return grid

def add_agents(grid, n_agents):
    agents = list(range(-n_agents,0))
    empty_indexes = np.stack(np.where(grid == 0)).T
    random_i = np.random.choice(range(empty_indexes.shape[0]), n_agents)
    agent_idxs = empty_indexes[random_i,:]
    for i, idx in enumerate(agent_idxs):
        grid[idx[0], idx[1]] = agents[i]
        
    return grid

def add_apples(grid, n_apples):
    n_agents = len(grid[grid<0])
    possible_values = list(range(1,min(8,n_agents+1)))
    empty_indexes = np.stack(np.where(grid == 0)).T
    random_i = np.random.choice(range(empty_indexes.shape[0]), n_apples)
    prize_idxs = empty_indexes[random_i,:]
    values = np.random.choice(possible_values, n_apples)
    for i, idx in enumerate(prize_idxs):
        grid[idx[0], idx[1]] = values[i]
        
    return grid

def build_grid(grid_size, n_agents, n_apples, astype=int):
    grid = build_empty_grid(grid_size)
    grid = add_agents(grid, n_agents)
    grid = add_apples(grid, n_apples)
    return grid.astype(astype)

# Function that plots the grid
def plot_game(grid):

    apple_locs = list(set(grid[grid>0]))
    agent_locs = list(set(grid[grid<0]))

    # Defining colors for each symbol
    colors = {0: 'lightgrey'}
    colors.update({loc:'red' for loc in apple_locs})
    colors.update({loc:'cyan' for loc in agent_locs})

    clear_output(wait=True)

    # Convert the matrix values to a list of colors
    unique_values = list(colors.keys())  # Extract unique values in the correct order
    
    color_list = [colors[val] for val in unique_values]  # Generate a list of colors in the order of unique values
    # Create a ListedColormap using the colors
    cmap = ListedColormap(color_list)
    
    # Create an index map to translate values in the matrix to their corresponding indices in the cmap
    index_map = np.vectorize(lambda x: unique_values.index(x))(grid)

    fig, ax = plt.subplots(1, figsize=(5,5))
    
    # Plot the matrix using imshow and the custom colormap
    ax.imshow(index_map, cmap=cmap)

    for (j,i),label in np.ndenumerate(grid):
        if label > 0:
            ax.text(i,j,label,ha='center',va='center', fontsize=14)
        elif label < 0:
            ax.text(i,j,r'$x_{}$'.format(label*-1), ha='center',va='center', fontsize=14)
    plt.axis('off')
