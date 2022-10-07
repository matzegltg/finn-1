import numpy as np
import torch as th
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
import time
import pickle
sys.path.append("..")
from utils.configuration import Configuration
from finn import *


def animate_1d(t, axis1, axis2, field, field_hat):
    """
    Data animation function animating an image over time.
    :param t: The current time step
    :param axis: The matplotlib image object
    :param field: The data field
    :return: The matplotlib image object updated with the current time step's
        image date
    """
    axis1.set_ydata(field[:, t])
    axis2.set_ydata(field_hat[:, t])

def init_model(number):
    with open(f"results/{number}/model.pkl", "rb") as inp:
        model = pickle.load(inp)
    
    u_hat = np.load(f"results/{number}/u_hat.npy")
    u = np.load(f"results/{number}/u.npy")    
    t = np.load(f"results/{number}/t.npy")
    x = np.load(f"results/{number}/x.npy")

    return model, u_hat, u, t, x

def vis_sol(model, u_hat, u, t, x):
    u_hat = np.transpose(u_hat[...,0])
    u = np.transpose(u[...,0])

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    # u(t, x) over space
    h = ax[0].imshow(u, interpolation='nearest', 
                    extent=[t.min(), t.max(),
                            x.min(), x.max()],
                    origin='upper', aspect='auto')
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)

    ax[0].set_xlim(0, t.max())
    ax[0].set_ylim(x.min(), x.max())
    ax[0].legend(loc="upper right")
    ax[0].set_xlabel('$t$')
    ax[0].set_ylabel('$x$')
    ax[0].set_title('FD: $c_w(t,x)$', fontsize = 10)

    l = ax[1].imshow(u_hat, interpolation='nearest', 
                    extent=[t.min(), t.max(),
                            x.min(), x.max()],
                    origin='upper', aspect='auto')
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(l, cax=cax)

    ax[1].set_xlim(0, t.max())
    ax[1].set_ylim(x.min(), x.max())
    ax[1].legend(loc="upper right")
    ax[1].set_xlabel('$t$')
    ax[1].set_ylabel('$x$')
    ax[1].set_title('Finn: $c_w(t,x)$', fontsize = 10)
    plt.show()

def vis_diff(model, u_hat, u, t, x):
    u_hat = np.transpose(u_hat[...,0])
    u = np.transpose(u[...,0])
    # diffplot

    diff = u_hat - u

    fig, ax = plt.subplots()

    # u(t, x) over space
    h = ax.imshow(diff, interpolation='nearest', 
                extent=[t.min(), t.max(),
                        x.min(), x.max()],
                origin='upper', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)

    ax.set_xlim(0, t.max())
    ax.set_ylim(x.min(), x.max())
    ax.legend(loc="upper right")
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.set_title('Difference of $c_w$ FD sol. vs FINN sol.', fontsize = 10)
    plt.show()

def vis_sorption(model, u_hat, u, t, x, number):

    params = Configuration(f"results/{number}/params/cfg.json")

    # exakt solution
    sk = np.transpose(u[...,1])
    cw = np.transpose(u[...,0])
    k_d = params.k_d
    beta = params.beta
    f = params.f
    se = f*k_d*cw**beta

    # approx FINN sol
    cw_hat = np.transpose(u_hat[...,0])
    sk_hat = np.transpose(u_hat[...,1])
    beta_hat = model.beta.item()
    k_d_hat = model.k_d.item()
    f_hat = model.f.item()
    se_hat = f_hat*k_d_hat*cw_hat**beta_hat

    fig, ax = plt.subplots(3,2)

    # plot s_k FD
    h = ax[0,0].imshow(sk, interpolation='nearest', 
                extent=[t.min(), t.max(),
                        x.min(), x.max()],
                origin='upper', aspect='auto')
    divider = make_axes_locatable(ax[0,0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    ax[0,0].set_xlim(0, t.max())
    ax[0,0].set_ylim(x.min(), x.max())
    ax[0,0].legend(loc="upper right")
    ax[0,0].set_xlabel('$t$')
    ax[0,0].set_ylabel('$x$')
    ax[0,0].set_title('FD: $s_k(t,x)$', fontsize = 10)
    
    # plot s_k FINN
    h = ax[0,1].imshow(sk_hat, interpolation='nearest', 
                extent=[t.min(), t.max(),
                        x.min(), x.max()],
                origin='upper', aspect='auto')
    divider = make_axes_locatable(ax[0,1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    ax[0,1].set_xlim(0, t.max())
    ax[0,1].set_ylim(x.min(), x.max())
    ax[0,1].legend(loc="upper right")
    ax[0,1].set_xlabel('$t$')
    ax[0,1].set_ylabel('$x$')
    ax[0,1].set_title('FINN: $s_k(t,x)$', fontsize = 10)

    # plot s_e FD
    h = ax[1,0].imshow(se, interpolation='nearest', 
                extent=[t.min(), t.max(),
                        x.min(), x.max()],
                origin='upper', aspect='auto')
    divider = make_axes_locatable(ax[1,0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    ax[1,0].set_xlim(0, t.max())
    ax[1,0].set_ylim(x.min(), x.max())
    ax[1,0].legend(loc="upper right")
    ax[1,0].set_xlabel('$t$')
    ax[1,0].set_ylabel('$x$')
    ax[1,0].set_title('FD: $s_e(t,x)$', fontsize = 10)

    # plot s_e FINN
    h = ax[1,1].imshow(se_hat, interpolation='nearest', 
                extent=[t.min(), t.max(),
                        x.min(), x.max()],
                origin='upper', aspect='auto')
    divider = make_axes_locatable(ax[1,1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    ax[1,1].set_xlim(0, t.max())
    ax[1,1].set_ylim(x.min(), x.max())
    ax[1,1].legend(loc="upper right")
    ax[1,1].set_xlabel('$t$')
    ax[1,1].set_ylabel('$x$')
    ax[1,1].set_title('FINN: $s_e(t,x)$', fontsize = 10)

    # plot c_w FD
    h = ax[2,0].imshow(cw, interpolation='nearest', 
                extent=[t.min(), t.max(),
                        x.min(), x.max()],
                origin='upper', aspect='auto')
    divider = make_axes_locatable(ax[2,0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    ax[2,0].set_xlim(0, t.max())
    ax[2,0].set_ylim(x.min(), x.max())
    ax[2,0].legend(loc="upper right")
    ax[2,0].set_xlabel('$t$')
    ax[2,0].set_ylabel('$x$')
    ax[2,0].set_title('FD: $c_w(t,x)$', fontsize = 10)

    # plot c_w FINN
    h = ax[2,1].imshow(cw_hat, interpolation='nearest', 
                extent=[t.min(), t.max(),
                        x.min(), x.max()],
                origin='upper', aspect='auto')
    divider = make_axes_locatable(ax[2,1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    ax[2,1].set_xlim(0, t.max())
    ax[2,1].set_ylim(x.min(), x.max())
    ax[2,1].legend(loc="upper right")
    ax[2,1].set_xlabel('$t$')
    ax[2,1].set_ylabel('$x$')
    ax[2,1].set_title('FINN: $c_w(t,x)$', fontsize = 10)
    plt.tight_layout(pad=0.2)
    plt.show()

def __vis_ret_old():
    fig, ax = plt.subplots()
    # u(t, x) over space
    h = ax.imshow(c-se, interpolation='nearest', 
                extent=[t.min(), t.max(),
                        x.min(), x.max()],
                origin='upper', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)

    ax.set_xlim(0, t.max())
    ax.set_ylim(x.min(), x.max())
    ax.legend(loc="upper right")
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.set_title('$c_w-s_e(t,x)$', fontsize = 10)

    # plot sorpt_isotherm
    fig, ax = plt.subplots(1,3)
    for timestep in range(0,len(t)):
        if timestep%500 == 0:
            ax[0].plot(c[:,timestep], sk[:,timestep], label=f"2ss: {timestep}")
    #frndl_rc = model.k_d*model.f*cs**model.beta
    #ax.plot(cs, frndl_rc, label="freundlich")
    #ax[0].legend()
    ax[0].set_xlabel("$c_w$")
    ax[0].set_ylabel("$s_k$")
    ax[0].set_title("sorption isotherm after #dt")

    for timestep in range(0,len(t)):
        if timestep%500 == 0:
            ax[1].plot(c[:,timestep], s_tot[:,timestep], label=f"2ss: {timestep}")
    #frndl_rc = model.k_d*model.f*cs**model.beta
    #ax.plot(cs, frndl_rc, label="freundlich")
    c_art = np.linspace(0,0.032,20)
    ax[1].plot(c_art, f*k_d*c_art, "--", label=f"linear instantaneous sorption")
    #ax[1].legend()
    ax[1].set_xlabel("$c_w$")
    ax[1].set_ylabel("$s_e + s_k$")
    ax[1].set_title("sorption isotherm after #dt")

    for timestep in range(0,len(t)):
        if timestep%500 == 0:
            ax[2].plot(c[:,timestep], se[:,timestep], label=f"2ss: {timestep}")
    #frndl_rc = model.k_d*model.f*cs**model.beta
    #ax.plot(cs, frndl_rc, label="freundlich")
    
    ax[2].plot(c_art, f*k_d*c_art, "--", label=f"linear instantaneous sorption")
    ax[2].legend(loc='lower left', bbox_to_anchor=(0.5, 1.05))
    ax[2].set_xlabel("$c_w$")
    ax[2].set_ylabel("$s_e$")
    ax[2].set_title("sorption isotherm after #dt")
    plt.show()

def __vis_ct_diff_old(model, u_hat, u, t, x):
    t = t.numpy()
    c_hat = np.transpose(u_hat[...,0])
    c = np.transpose(u[...,0])
    c_hat_t = np.transpose(u_hat[...,1])
    c_t = np.transpose(u[...,1])
    n_e = model.n_e.item()
    rho_s = model.rho_s.item()
    beta = model.beta.item()
    k_d = model.k_d.item()
    f =model.f.item()
    sk  = (c_t-c*n_e -rho_s*(c**beta*k_d)*f)/(rho_s)
    se = f*k_d*c**beta
    s_tot = se+sk
    c_tot_calc = s_tot*rho_s + n_e*c
    fig, ax = plt.subplots()
    # u(t, x) over space
    h = ax.imshow(c_t - c_tot_calc, interpolation='nearest', 
                extent=[t.min(), t.max(),
                        x.min(), x.max()],
                origin='upper', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)

    ax.set_xlim(0, t.max())
    ax.set_ylim(x.min(), x.max())
    ax.legend(loc="upper right")
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.set_title('$c_t - c_tot_calc$', fontsize = 10)

    plt.show()

def comp_ret_soil(model, u_hat, u, t, x, number):

    params = Configuration(f"results/{number}/params/cfg.json")

    # exakt solution
    sk = np.transpose(u[...,1])
    cw = np.transpose(u[...,0])

    if params.sandbool:
        sk = sk[params.sand.top:params.sand.bot]
        cw = cw[params.sand.top:params.sand.bot]

    k_d = params.k_d
    beta = params.beta
    f = params.f
    se = f*k_d*cw**beta

    # approx FINN sol
    cw_hat = np.transpose(u_hat[...,0])
    sk_hat = np.transpose(u_hat[...,1])

    if params.sandbool:
        cw_hat = cw_hat[params.sand.top:params.sand.bot]
        sk_hat = sk_hat[params.sand.top:params.sand.bot]

    beta_hat = model.beta.item()
    k_d_hat = model.k_d.item()
    f_hat = model.f.item()
    se_hat = f_hat*k_d_hat*cw_hat**beta_hat


    fig, ax = plt.subplots(1,2)
    # plot sk over cw
    for timestep in range(0,len(t)):
        if timestep%500 == 0:
            ax[0].plot(cw[:,timestep], sk[:,timestep], label=f"FD: {timestep}", alpha=0.9)
            ax[0].plot(cw_hat[:, timestep], sk_hat[:,timestep], "--", label=f"FINN: {timestep}", alpha=0.9)

    ax[0].set_xlabel("$c_w$")
    ax[0].set_ylabel("$s_k$")
    ax[0].set_title("kin. sorb. isotherms (after #dt)")
    ax[0].legend()
    
    # plot se over cw
    for timestep in range(0,len(t)):
        if timestep%500 == 0:
            ax[1].plot(cw[:,timestep], se[:,timestep], label=f"FD: {timestep}")
            ax[1].plot(cw_hat[:, timestep], se_hat[:,timestep], "--", label=f"FINN: {timestep}")
    
    #ax[1].legend()
    ax[1].set_xlabel("$c_w$")
    ax[1].set_ylabel("$s_e$")
    ax[1].set_title("inst. sorb. isotherms (after #dt)")
    ax[1].legend()
    plt.show()

def load_model(number):
    model, u_hat, u, t, x = init_model(number)
    print(model.__dict__)
    vis_sol(model, u_hat, u, t, x)
    vis_diff(model, u_hat, u, t, x)
    vis_sorption(model, u_hat, u, t,x, number)
    comp_ret_soil(model, u_hat, u, t, x, number)


if __name__ == "__main__":
    load_model(number=7)

# Method to vis sk/se over cw inclusive sand -> anything to see?
def __comp_ret_total(model, u_hat, u, t, x, number):

    params = Configuration(f"results/{number}/params/cfg.json")

    # exakt solution
    sk = np.transpose(u[...,1])
    cw = np.transpose(u[...,0])
    k_d = params.k_d
    beta = params.beta
    f = params.f
    se = f*k_d*cw**beta

    # approx FINN sol
    cw_hat = np.transpose(u_hat[...,0])
    sk_hat = np.transpose(u_hat[...,1])
    beta_hat = model.beta.item()
    k_d_hat = model.k_d.item()
    f_hat = model.f.item()
    se_hat = f_hat*k_d_hat*cw_hat**beta_hat


    fig, ax = plt.subplots(1,2)
    # plot sk over cw
    for timestep in range(0,len(t)):
        if timestep%500 == 0:
            print(cw[:,timestep].shape)
            print(sk[:,timestep].shape)
            ax[0].scatter(cw[:,timestep], sk[:,timestep], label=f"FD: {timestep}")
            ax[0].scatter(cw_hat[:, timestep], sk_hat[:,timestep], label=f"FINN: {timestep}")

    ax[0].set_xlabel("$c_w$")
    ax[0].set_ylabel("$s_k$")
    ax[0].set_title("kin. sorb. isotherms (after #dt)")
    ax[0].legend()
    
    # plot se over cw
    for timestep in range(0,len(t)):
        if timestep%500 == 0:
            ax[1].plot(cw[:,timestep], se[:,timestep], label=f"FD: {timestep}")
            ax[1].plot(cw_hat[:, timestep], se_hat[:,timestep], "--", label=f"FINN: {timestep}")
    
    #ax[1].legend()
    ax[1].set_xlabel("$c_w$")
    ax[1].set_ylabel("$s_e$")
    ax[1].set_title("inst. sorb. isotherms (after #dt)")
    ax[1].legend()
    plt.show()
