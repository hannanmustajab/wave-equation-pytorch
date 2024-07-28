import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
from Common import NeuralNet
import time
torch.manual_seed(128)
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
import scienceplots
import os
import logging
plt.style.use(['science','no-latex','grid','bright','ieee'])
from matplotlib.colors import LogNorm

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    print ("MPS device not found.")

    
logging.getLogger('matplotlib.font_manager').disabled = True



def create_fdm_comparison_plots(fdm_file_path, pinn_model, num_time_steps, c):
    """
    Create comparison plots between FDM and PINN solutions and calculate L2 error.

    Parameters:
    - fdm_file_path (str): File path to the FDM data file (npz format).
    - pinn_model: The trained PINN model.
    - num_time_steps (int): Number of time steps to extract from the FDM model.
    - c: Velocity parameter.
    """

    # Load FDM data
    fdm = np.load(fdm_file_path)
    x_fdm, t_fdm, U_fdm = fdm['X'], fdm['T'], fdm['U']

    # Since the size of fdm tensor is huge, select only 100 time steps on which we will compare both the results. 
    # Determine the indices for the selected time steps
    selected_indices = np.round(np.linspace(0, len(t_fdm) - 1000, num_time_steps)).astype(int)

    # Extract selected time steps and corresponding solution
    selected_time_steps = t_fdm[selected_indices]
    # selected_time_steps = selected_time_steps[:-1]
    selected_solution = U_fdm[:, selected_indices]

    # Print the shape of each array
    print("Shape of x_fdm:", x_fdm.shape)
    print("Shape of selected_time_steps:", selected_time_steps.shape)
    print("Shape of selected_solution:", selected_solution.shape)

    # Flatten the arrays for scatter plot
    flat_x = x_fdm.reshape(-1, 1).repeat(len(selected_time_steps), axis=1).flatten()
    flat_t = np.tile(selected_time_steps, len(x_fdm))
    flat_u = selected_solution.flatten()
    
    # Meshgrid for scatter plot
    X_input = np.meshgrid(selected_time_steps, x_fdm)

    # Reshape the meshgrid to a 2D array
    X_input = np.column_stack((X_input[0].flatten(), X_input[1].flatten()))

    # Print the shape of X_input
    print("Shape of X_input:", X_input.shape)
    
    # Compute PINN solution on the FDM data points. [t,x]
    pinn_sol_fdm_data = pinn_model.approximate_solution(torch.tensor(X_input,dtype=torch.float32).to(device))
    pinn_sol_fdm_data = pinn_sol_fdm_data.detach().cpu().reshape(-1,)
    
    # Create folders if they don't exist
    folder_path = f'fdm_comp/c_{c}'
    os.makedirs(folder_path, exist_ok=True)

    # Scatter plot for FDM solution
    plt.figure(figsize=(5, 10))
    plt.scatter(flat_x, flat_t, c=flat_u, cmap='jet', antialiased=True)
    plt.colorbar(label='FDM Solution Value')
    plt.title('FDM Solution')
    plt.xlabel('Spatial Coordinate (x)')
    plt.ylabel('Time (t)')
    plt.xlim([0, 3.14])
    plt.ylim([0, 2 * 3.14])
    plt.savefig(os.path.join(folder_path, 'fdm_solution.png'), dpi=600)
    plt.close()

    # Scatter plot for PINN solution
    plt.figure(figsize=(5, 10))
    plt.scatter(flat_x, flat_t, c=pinn_sol_fdm_data, cmap='jet', antialiased=True)
    plt.colorbar(label='PINN Solution Value')
    plt.title('PINN Solution')
    plt.xlabel('Spatial Coordinate (x)')
    plt.ylabel('Time (t)')
    plt.xlim([0, 3.14])
    plt.ylim([0, 2 * 3.14])
    plt.savefig(os.path.join(folder_path, 'pinn_solution.png'), dpi=600)
    plt.close()

    # Scatter plot for the difference with colorbar limit
    plt.figure(figsize=(5, 10))
    difference_scatter = plt.scatter(flat_x, flat_t, c=flat_u - pinn_sol_fdm_data.numpy(), cmap='jet', antialiased=True, vmin=-0.01, vmax=0.01)
    cbar = plt.colorbar(difference_scatter, label='Difference')
#     cbar.set_clim(-0.01, 0.01)  # Adjust the color limits as needed
    plt.title('Difference between FDM and PINN Solution')
    plt.xlabel('Spatial Coordinate (x)')
    plt.ylabel('Time (t)')
    plt.xlim([0, 3.14])
    plt.ylim([0, 2 * 3])
    plt.savefig(os.path.join(folder_path, 'difference.png'), dpi=600)
    plt.close()

    # Calculate L2 error
    err = (np.mean((pinn_sol_fdm_data.numpy() - flat_u) ** 2) / np.mean(flat_u ** 2)) ** 0.5 * 100
    print(f'L2 Relative Error Norm for c={c}: {err:.2f}%')

    # Save FDM and PINN solutions, and the difference
    np.savez(os.path.join(folder_path, 'fdm_data.npz'), x=flat_x, t=flat_t, u=flat_u)
    np.savez(os.path.join(folder_path, 'pinn_data.npz'), x=flat_x, t=flat_t, u=pinn_sol_fdm_data.numpy())
    np.savez(os.path.join(folder_path, 'difference_data.npz'), x=flat_x, t=flat_t, difference=flat_u - pinn_sol_fdm_data.numpy())
