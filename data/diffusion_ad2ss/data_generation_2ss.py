# @File          :   data_generation_2ss.py
# @Last modified :   2022/09/12 11:19:05
# @Author        :   Matthias Gueltig

"""
This script is a modified version of DS solver. To solve PDE that describes PFAS
transport through soil
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
from simulator_2ss import Simulator
import os
import sys

sys.path.append("..")
from utils.configuration import Configuration

# general parameters
TRAIN_DATA = True
DATASET_NAME = "data"

DATAPOINTS_INITIAL = 15
DATAPOINTS_BOUNDARY = 50
DATAPOINTS_COLLOCATION = 10000

def read_exp_velocities():
    # cm/d
    # get velocities
    df = pd.read_excel("Initials.xlsx", "hydraulics", skiprows=4, nrows=21, usecols="B:F")
    v = df["cm/d"].to_numpy()
    v = np.delete(v, 0)

    meas_day = df["sampling_ time [days]"].to_numpy()
    meas_day = np.delete(meas_day, 0)
    return v, meas_day
    
def generate_sample(simulator, visualize, save_data, root_path):
    """
    This function generates a data sample, visualizes it if desired and saves
    the data to file if desired.
    :param simulator: The simulator object for data creation
    :param visualize: Boolean indicating whether to visualize the data
    :param save_Data: Boolean indicating whether to write the data to file
    :param root_path: The root path of this script
    """

    print("Generating data...")

    # Generate a data sample
    sample_c, sample_sk = simulator.generate_sample()
    
    
    if TRAIN_DATA:

        # Randomly draw indices for initial, boundary and collocation points
        #idcs_init, idcs_bound = draw_indices(
        #    simulator=simulator,
        #    n_init=DATAPOINTS_INITIAL,
        #    n_bound=DATAPOINTS_BOUNDARY,
        #    n_colloc=DATAPOINTS_COLLOCATION
        #)
        
    
        if visualize:
            
            visualize_sample(sample=sample_c, simulator=simulator)
            visualize_sample(sample=sample_sk, simulator=simulator)
        
        if save_data:
            write_data_to_file(
                root_path=root_path,
                simulator=simulator,
                sample_c=sample_c,
                sample_sk=sample_sk
            )
            # List for tuples as train/val/test data
            data_tuples = []

def write_data_to_file(root_path, simulator, sample_c, sample_sk):
    """
    Writes the given data to the according directory in .npy format.
    :param root_path: The root_path of the script
    :param simulator: The simulator that created the data
    :param sample_c: The sample to be written to file (dissolved concentration)
    :param sample_sk: The sample to be written to file (kin. sorbed concentration)
    """
    
    if TRAIN_DATA:
    
        # Create the data directory for the training data if it does not yet exist
        data_path = os.path.join(root_path, DATASET_NAME+"_train")
        os.makedirs(data_path, exist_ok=True)
        
        # Write the t- and x-series data along with the sample to file
        np.save(file=os.path.join(data_path, "t_series.npy"),
                arr=simulator.t[:len(simulator.t)//4 + 1])
        
        np.save(file=os.path.join(data_path, "x_series.npy"), arr=simulator.x)
        np.save(file=os.path.join(data_path, "sample_c.npy"),
                arr=sample_c[:len(simulator.t)//4 + 1])
        np.save(file=os.path.join(data_path, "sample_sk.npy"),
                arr=sample_sk[:len(simulator.t)//4 + 1])
            
        # Create the data directory for the extrapolation data if it does not yet exist
        data_path = os.path.join(root_path, DATASET_NAME+"_ext")
        os.makedirs(data_path, exist_ok=True)
        
        # Write the t- and x-series data along with the sample to file
        np.save(file=os.path.join(data_path, "t_series.npy"), arr=simulator.t)
        np.save(file=os.path.join(data_path, "x_series.npy"), arr=simulator.x)
        np.save(file=os.path.join(data_path, "sample_c.npy"), arr=sample_c)
        np.save(file=os.path.join(data_path, "sample_sk.npy"), arr=sample_sk)
    
    else:

        # Create the data directory if it does not yet exist
        data_path = os.path.join(root_path, DATASET_NAME+"_test")
        os.makedirs(data_path, exist_ok=True)
    
        # Write the t- and x-series data along with the sample to file
        np.save(file=os.path.join(data_path, "t_series.npy"), arr=simulator.t)
        np.save(file=os.path.join(data_path, "x_series.npy"), arr=simulator.x)
        np.save(file=os.path.join(data_path, "sample_c.npy"), arr=sample_c)
        np.save(file=os.path.join(data_path, "sample_sk.npy"), arr=sample_sk)
    
def draw_indices(simulator, n_init, n_bound, n_colloc):
    """
    Randomly chooses a specified number of points from the spatiotemporal
    sample for the initial and boundary conditions as well as collocation
    points.
    :param simulator: The simulator that created the sample
    :param n_init: Number of initial points at t=0
    :param n_bound: Number of boundary points at x_left and x_right
    :param n_colloc: Number of collocation points
    :return: The two-dimensional index arrays(t, x)
    """

    rng = np.random.default_rng()

    idcs_init = np.zeros((n_init, 2), dtype=np.int)
    idcs_init[:, 0] = 0
    idcs_init[:, 1] = rng.choice(len(simulator.x),
                                 size=n_init,
                                 replace=False)

    idcs_bound = np.zeros((n_bound, 2), dtype=np.int)
    idcs_bound[:n_bound//2, 0] = rng.choice(len(simulator.t)//4 + 1,
                                  size=n_bound//2,
                                  replace=False)
    idcs_bound[:n_bound//2, 1] = 0
    
    idcs_bound[n_bound//2:, 0] = rng.choice(len(simulator.t)//4 + 1,
                                  size=n_bound - n_bound//2,
                                  replace=False)
    idcs_bound[n_bound//2:, 1] = len(simulator.x) - 1

    return idcs_init, idcs_bound

def visualize_sample(sample, simulator):
    """
    Method to visualize a single sample. Code taken and modified from
    https://github.com/maziarraissi/PINNs
    :param sample: The actual data sample for visualization
    :param simulator: The simulator used for data generation
    :param idcs_init: The indices of the initial points
    :param idcs_bound: The indices of the boundary points
    """

    sample = np.transpose(sample)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    # u(t, x) over space
    h = ax[0].imshow(sample, interpolation='nearest', cmap='rainbow', 
                  extent=[simulator.t.min(), simulator.t.max(),
                          simulator.x.min(), simulator.x.max()],
                  origin='upper', aspect='auto')
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
        
    ax[0].set_xlim(0, simulator.t.max())
    ax[0].set_ylim(simulator.x.min(), simulator.x.max())
    ax[0].legend(loc="upper right")
    ax[0].set_xlabel(r'$t [d]$')
    ax[0].set_ylabel(r'$x [cm]$')
    ax[0].set_title(r'$u(t,x) [\frac{\mu g}{cm^3}]$', fontsize = 10)
    
    # u(t, x) over time
    line, = ax[1].plot(simulator.x, sample[:, 0], 'b-', linewidth=2, label='Exact')
    ax[1].set_xlabel('$x$')
    ax[1].set_ylabel('$u(t,x)$')    
    ax[1].set_xlim([simulator.x.min(), simulator.x.max()])
    ax[1].set_ylim([0, 1.1])

    anim = animation.FuncAnimation(fig,
                                   animate,
                                   frames=len(simulator.t),
                                   fargs=(line, sample),
                                   interval=20)
    plt.tight_layout()
    plt.draw()
    
    plt.show()

def write_tuples_to_file(root_path, data_tuples, mode):
    """
    Writes the data tuples to the according directory in .npy format for
    training and validation of PINN
    :param root_path: The root_path of the script
    :param data_tuples: Array of the train/val tuples
    :param mode: Any of "train" or "val"
    
    """
    
    data_path = os.path.join(root_path, DATASET_NAME+"_train")
    os.makedirs(os.path.join(data_path, mode), exist_ok=True)
    
    # Iterate over the data_tuples and write them to separate files
    for idx, data_tuple in enumerate(data_tuples):
        
        name = f"{mode}_{str(idx).zfill(5)}.npy"
        np.save(file=os.path.join(data_path, mode, name), arr=data_tuple)

def create_data_tuple_init_bound(simulator, sample_c, sample_ct, pair):
    """
    Creates a tuple (t, x, sample_c, sample_ct, t_idx, x_idx), where t is the
    time step, x is the spatial coordinate, sample_c and sample_ct are the 
    desired model output, and t_idx and x_idx are the indices in the sample for t and x.
    :param simulator: The simulator that generated the sample
    :param sample_c: The data sample (dissolved concentration)
    :param sample_ct: The data sample (total concentration)
    :param pair: The index pair of the current data points
    
    :return: Tuple (t, x, sample_c, sample_ct, t_idx, x_idx)
    """
    t_idx, x_idx = pair
    c = sample_c[t_idx, x_idx]
    ct = sample_ct[t_idx, x_idx]
    t, x = simulator.t[t_idx], simulator.x[x_idx]
    
    return np.array((t, x, c, ct, t_idx, x_idx), dtype=np.float32)

def create_data_tuple_colloc(simulator, sample_c, sample_ct):
    """
    Creates a tuple (t, x, sample_c, sample_ct, t_idx, x_idx), where t is the
    time step, x is the spatial coordinate, sample_c and sample_ct are the 
    desired model output, and t_idx and x_idx are the indices in the sample for t and x.
    :param simulator: The simulator that generated the sample
    :param sample_c: The data sample (dissolved concentration)
    :param sample_ct: The data sample (total concentration)
    
    :return: Tuple (t, x, sample_c, sample_ct, t_idx, x_idx)
    """
    t = np.arange(len(simulator.t)//2 + 1)
    x = np.arange(len(simulator.x))
    
    t, x = np.meshgrid(t,x)
    
    pair = np.hstack((t.flatten()[:,None],x.flatten()[:,None]))
    idx = np.random.choice(len(pair), DATAPOINTS_COLLOCATION , replace=False)
    
    t_idx = pair[idx,0]
    x_idx = pair[idx,1]
    
    c = sample_c[t_idx, x_idx]
    ct = sample_ct[t_idx, x_idx]
    
    t, x = simulator.t[t_idx], simulator.x[x_idx]
    
    return np.array((t, x, c, ct, t_idx, x_idx), dtype=np.float32)

def animate(t, axis, field):
    """
    Data animation function animating an image over time.
    :param t: The current time step
    :param axis: The matplotlib image object
    :param field: The data field
    :return: The matplotlib image object updated with the current time step's
        image date
    """
    axis.set_ydata(field[:, t])
  

def main():
    """
    Main method used to create the datasets.
    """
    ##############
    # PARAMETERS #
    ##############
    
    # Determine the root path for this script and set up a path for the data
    root_path = os.path.abspath("")
    
    params = Configuration("params.json")

    T_STEPS = params.T_STEPS
    X_STEPS = params.X_STEPS

    # read experimental velocities 
    exp_velocities = False

    if exp_velocities:
        v, meas_day = read_exp_velocities()
    else:
        # velocity = q/n_e
        v = params.v_e
        meas_day = 1

    # mug/l
    init_conc = params.init_conc
    s_k_0 = params.kin_sorb

    SAVE_DATA = True
    VISUALIZE_DATA = True
    
    if params.sand.bool:
        simulator = Simulator(
            d_e=params.D_e,
            n_e=params.porosity,
            rho_s=params.rho_s,
            beta=params.beta,
            f=params.f,
            k_d=params.k_d,
            cw_0=init_conc,
            t_max=params.T_MAX,
            t_steps=params.T_STEPS,
            x_left=params.X_LEFT,
            x_right=params.X_RIGHT,
            x_steps=X_STEPS,
            v=v,
            a_k=params.a_k,
            m_soil=params.m_soil,
            alpha_l = params.alpha_l,
            s_k_0 = s_k_0,
            n_e_sand = params.sand.porosity,
            x_start_soil= params.sand.top,
            x_stop_soil = params.sand.bot,
            x_steps_sand=params.sand.X_STEPS,
            alpha_l_sand = params.sand.alpha_l,
            v_e_sand = params.sand.v_e
            )

    else:
        # Create a wave generator using the parameters from the configuration file
        simulator = Simulator(d_e=params.D_e,
        exp_v = exp_velocities,
        n_e=params.porosity,
        meas_day=meas_day,
        rho_s=params.rho_s,
        beta=params.beta,
        f=params.f,
        k_d=params.k_d,
        cw_0=init_conc,
        t_max=params.T_MAX,
        t_steps=params.T_STEPS,
        x_left=params.X_LEFT,
        x_right=params.X_RIGHT,
        x_steps=X_STEPS,
        v=v,
        a_k=params.a_k,
        m_soil=params.m_soil,
        alpha_l = params.alpha_l,
        s_k_0 = s_k_0)

    # Create train, validation and test data
    generate_sample(simulator=simulator,
                    visualize=VISUALIZE_DATA,
                    save_data=SAVE_DATA,
                    root_path=root_path)
    

if __name__ == "__main__":
    main()

    print("Done.")
