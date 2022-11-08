# @File          :   data_generation_2ss.py
# @Last modified :   2022/09/12 11:19:05
# @Author        :   Matthias Gueltig

"""
This script is a modified version of DS solver. To solve PDE that describes PFOS
transport through soil
"""

import numpy as np
import matplotlib.pyplot as plt
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

        if visualize:
            visualize_sample(sample_c=sample_c, sample_sk=sample_sk, simulator=simulator)

        if save_data:
            write_data_to_file(
                root_path=root_path,
                simulator=simulator,
                sample_c=sample_c,
                sample_sk=sample_sk
            )

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
        np.save(file="../../../../../OneDrive - bwedu/6. Semester/BA/finn_self/comp_adapt_bound/t_series.npy", arr=simulator.t)
        np.save(file="../../../../../OneDrive - bwedu/6. Semester/BA/finn_self/comp_adapt_bound/sample_c.npy", arr=sample_c)
        np.save(file="../../../../../OneDrive - bwedu/6. Semester/BA/finn_self/comp_adapt_bound/sample_sk.npy", arr=sample_sk)
        
    
    else:

        # Create the data directory if it does not yet exist
        data_path = os.path.join(root_path, DATASET_NAME+"_test")
        os.makedirs(data_path, exist_ok=True)
    
        # Write the t- and x-series data along with the sample to file
        np.save(file=os.path.join(data_path, "t_series.npy"), arr=simulator.t)
        np.save(file=os.path.join(data_path, "x_series.npy"), arr=simulator.x)
        np.save(file=os.path.join(data_path, "sample_c.npy"), arr=sample_c)
        np.save(file=os.path.join(data_path, "sample_sk.npy"), arr=sample_sk)

def visualize_sample(sample_c, sample_sk, simulator):
    """
    Method to visualize a single sample. Code taken and modified from
    https://github.com/maziarraissi/PINNs
    :param sample_c: The actual data sample for visualization
    .param sample_sk: The kin. sorb. data sample for visualization
    :param simulator: The simulator used for data generation
    """
    fig, ax = plt.subplots(1,2, figsize=(10, 4))
    
    sample_c = np.transpose(sample_c)
    
    # c_w(t, x) over space and time
    h = ax[0].imshow(sample_c, interpolation='nearest', cmap='rainbow', 
                  extent=[simulator.t.min(), simulator.t.max(),
                          simulator.x.min(), simulator.x.max()],
                  origin='upper', aspect='auto')
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
        
    ax[0].set_xlim(0, simulator.t.max())
    ax[0].set_ylim(simulator.x.min(), simulator.x.max())
    ax[0].legend(loc="upper right", fontsize = 17)
    ax[0].set_xlabel(r'$t [d]$', fontsize = 17)
    ax[0].set_ylabel(r'$x [cm]$', fontsize = 17)
    ax[0].set_title(r'$c_w(t,x) \left[\frac{\mu g}{cm^3}\right]$', fontsize = 17)
    ax[0].tick_params(axis='x', labelsize=17)
    ax[0].tick_params(axis='y', labelsize=17)
    plt.yticks(fontsize=17)
    # s_k(t,x) over space and time
    sample_sk = np.transpose(sample_sk)

    h = ax[1].imshow(sample_sk, interpolation='nearest', cmap='rainbow', 
                extent=[simulator.t.min(), simulator.t.max(),
                        simulator.x.min(), simulator.x.max()],
                origin='upper', aspect='auto')
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
        
    ax[1].set_xlim(0, simulator.t.max())
    ax[1].set_ylim(simulator.x.min(), simulator.x.max())
    ax[1].legend(loc="upper right", fontsize = 17)
    ax[1].set_xlabel(r'$t [d]$', fontsize = 17)
    ax[1].set_title(r'$s_k(t,x) \left[\frac{\mu g}{g}\right]$', fontsize = 17)
    ax[1].tick_params(axis='x', labelsize=17)
    ax[1].tick_params(axis='y', labelsize=17)
    plt.tight_layout()
    plt.yticks(fontsize=17)
    
    plt.show()

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

    # velocity = q/n_e
    v = params.v_e

    SAVE_DATA = True
    VISUALIZE_DATA = True
    sand = params.sandbool
    if sand:
        simulator = Simulator(
            d_e=params.D_e,
            n_e=params.porosity,
            rho_s=params.rho_s,
            beta=params.beta,
            f=params.f,
            k_d=params.k_d,
            cw_0=params.init_conc,
            t_max=params.T_MAX,
            t_steps=params.T_STEPS,
            x_right=params.X_LENGTH,
            x_steps=params.X_STEPS,
            v=v,
            a_k=params.a_k,
            alpha_l = params.alpha_l,
            s_k_0 = params.kin_sorb,
            sand=params.sandbool,
            n_e_sand = params.sand.porosity,
            x_start_soil= params.sand.top,
            x_stop_soil = params.sand.bot,
            alpha_l_sand = params.sand.alpha_l,
            v_e_sand = params.sand.v_e
            )

    else:
        simulator = Simulator(
            d_e=params.D_e,
            n_e=params.porosity,
            rho_s=params.rho_s,
            beta=params.beta,
            f=params.f,
            k_d=params.k_d,
            cw_0=params.init_conc,
            t_max=params.T_MAX,
            t_steps=params.T_STEPS,
            x_right=params.X_LENGTH,
            x_steps=params.X_STEPS,
            v=v,
            a_k=params.a_k,
            alpha_l=params.alpha_l,
            s_k_0 = params.kin_sorb,
            sand=params.sandbool
        )

    # Create train, validation and test data
    generate_sample(simulator=simulator,
                    visualize=VISUALIZE_DATA,
                    save_data=SAVE_DATA,
                    root_path=root_path)
    

if __name__ == "__main__":
    main()

    print("Done.")
