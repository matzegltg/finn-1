# @File          :   simulator_2ss.py
# @Last modified :   2022/10/05 17:01:28
# @Author        :   Matthias Gueltig

import numpy as np
import time
import sys

class Simulator(object):
    """Solves Advection-Diffusion equation for 2 side sorption.
    """
    def __init__(self, d_e:float, n_e:float, rho_s:float, beta:float, f:float, 
        k_d:float, cw_0:float, t_max:float, x_right:float, x_steps:int, 
        t_steps:int, v:float, a_k:float, alpha_l:float, s_k_0:float, sand:bool,
        n_e_sand:float=None, x_start_soil:float=None, x_stop_soil:float=None, 
        x_steps_sand:float=None, alpha_l_sand:float=None, v_e_sand:float=None):
        """Constructor method initializing the parameters for the diffusion
        sorption problem.

        Args:
            d_e (float): molecular diffusion coeff. [L^3/T]
            n_e (float): effective porosity [-]
            rho_s (float): dry bulk density of soil [M/L^3]
            beta (float): parameter for sorption isotherms [-]
            f (float): coupling kinetic and instantaneous sorption [-]
            k_d (float): parameter of sorption isotherm [L/M]
            cw_0 (float): initial dissolved concentration [M/L]
            t_max (float): end time of simulation [T]
            x_right (float): right end of the 1D simulation field
            x_steps (int): number of spatial steps between 0 and x_right 
            t_steps (int): number of time steps
            v (float): advective velocity [L/T]
            a_k (float): first order rate constant of eq 17c Simunek et al. [1/T]
            alpha_l (float): longitudinal dispersion coefficient [L]
            s_k_0 (float): init. mass conc. of kinetically sorbed PFOS [M/M]
            sand (bool): True if sand data exists
            n_e_sand (float, optional): Porosity of sand [-]
            x_start_soil (float, optional): spatial index at which sand turns into soil
            x_stop_soil (float, optional): spatial index at which soil turns back to sand
            x_steps_sand (float, optional): Number of nodes that should be considered as sand
            alpha_l_sand (float, optional): longitudinal dispersion coefficient in sand [L]
            v_e_sand (float, optional): advective velocity in sand [L/T]
        """ 
        # set class parameters
        self.n_e = n_e
        self.beta = beta
        self.f = f
        self.k_d = k_d
        self.cw_0 = cw_0
        self.s_k_0 = s_k_0
        self.t_max = t_max
        self.x_right = x_right
        self.x_steps = x_steps
        self.a_k = a_k
        self.t_steps = t_steps
        self.rho_s = rho_s
        self.sand = sand

        if self.sand:
            self.n_e_sand = n_e_sand
            self.x_steps_sand = x_steps_sand
            self.x_start = x_start_soil
            self.x_stop = x_stop_soil

            self.v = np.ndarray(self.x_steps)
            self.v[:self.x_start] = v_e_sand
            self.v[self.x_start:self.x_stop] = v
            self.v[self.x_stop:] = v_e_sand

            # no molecular diff in sand
            disp_sand = v_e_sand*alpha_l_sand
            disp_soil = v*alpha_l+d_e

            self.disp = np.ndarray(self.x_steps)
            self.disp[:self.x_start] = disp_sand
            self.disp[self.x_start:self.x_stop] = disp_soil
            self.disp[self.x_stop:] = disp_sand

        else:
            self.v = np.ones(self.x_steps)*v
            self.disp = self.v*alpha_l+d_e
            
        # consider x_right cells cells
        self.x = np.linspace(0, self.x_right, self.x_steps)
        self.dx = self.x[1] - self.x[0]

        self.t = np.linspace(0, self.t_max, self.t_steps)
        self.dt = self.t[1] -self.t[0]
        
        # in this Bachelor Thesis, Freundlich sorption is considered
        self.sorpt_isotherm = self.freundlich
        self.sorpt_derivat = self.d_freundlich


    def freundlich(self, c):
        """implements freundlich sorpotion sorption isotherm with K_d [M/M], beta, c[M/L^3]"""
        # numpy roots of negative numbers? 
        # see https://stackoverflow.com/questions/45384602/numpy-runtimewarning
        # -invalid-value-encountered-in-power
        return np.sign(c) * (np.abs(c))**self.beta*self.k_d
    
    def d_freundlich(self, c):
        """returns derivation of freundlich sorption isotherm [M/TL^3]"""
        #return derivative(self.linear, c, dx=1e-16)
        return np.sign(c)*np.abs(c)**(self.beta-1)*self.beta*self.k_d


    def generate_sample(self):
        """Function that generates solution for PDE problem.
        """
        # Laplacian matrix for diffusion term
        nx = np.diag(-2*np.ones(self.x_steps), k=0)
        nx_minus_1 = np.diag(np.ones(self.x_steps-1), k=-1)
        nx_plus_1 = np.diag(np.ones(self.x_steps-1), k=1)
        self.lap = nx + nx_minus_1 + nx_plus_1
        self.lap /= self.dx**2

        # symmetric differences for advection term
        nx_fd = np.diag(np.ones(self.x_steps), k=0)
        nx_fd_plus_1 = np.diag(np.ones(self.x_steps-1)*(-1), k=-1)
        self.fd = nx_fd + nx_fd_plus_1
        self.fd /= self.dx

        # solution vector with c_w in first self.x_steps rows and s_k in last
        # self.x_step columns 
        u = np.ndarray((2*self.x_steps, len(self.t)))
        u [:,0] = np.zeros(2*self.x_steps)
        if self.sand:
            u[self.x_start:self.x_stop, 0] = self.cw_0
            u[self.x_steps+self.x_start:self.x_steps+self.x_stop,0] = self.s_k_0
        else:
            u[:self.x_steps,0] = self.cw_0
            u[self.x_steps:,0] = self.s_k_0
        sol = self.solve_ivp_euler_exp(u=u)
        
        # sample_c: [timesteps, spatial steps], dissolved conc.
        # sample_sk: [timesteps, spatial steps], kin. sorbec conc.
        sample_c = np.transpose(sol[:self.x_steps,:])
        sample_sk = np.transpose(sol[self.x_steps:,:])

        return sample_c, sample_sk

    def solve_ivp_euler_exp(self, u):
        """simple explicit euler to integrate ode"""
        for i in range(len(self.t)-1):
            # To observe solution status
            if i%10000 == 0:
                print(i)
            u[:,i+1] = u[:,i] + self.dt*self.ad_ode(t=i, conc_cw_sk=u[:,i])

        return u

    def ad_ode(self, t, conc_cw_sk:np.ndarray):
        """function that should be integrated over time

        Args:
            t (time): timestep
            conc_cw_sk (np.ndarray): concatenated c_w and s_k
        """

        # split u in cw and sk
        cw = conc_cw_sk[:self.x_steps]
        sk = conc_cw_sk[self.x_steps:]
        

        # setups boundarys for cw which are not accesed by fd and lap
        # in case nothing else is needed put zeros in the array
        # top dirichlet boundary
        dif_bound = np.zeros(self.x_steps)
        dif_bound[0] = -self.v[0]/self.dx*0
        dif_bound[-1] = self.disp[-1]/(self.dx**2)*cw[-1]

        if self.sand:
            sk_soil = sk[self.x_start:self.x_stop]
            cw_soil = cw[self.x_start:self.x_stop]

            inhomog = np.zeros(self.x_steps)
            div = np.ones(self.x_steps)

            # create inhomogenous and divisor values for soil area
            inhomog[self.x_start:self.x_stop] = self.a_k * \
                ((1-self.f)*self.sorpt_isotherm(cw_soil)-sk_soil)
            div[self.x_start:self.x_stop] = self.f*self.sorpt_derivat(cw_soil)* \
                (self.rho_s/self.n_e) + 1

            # calculate change of cw and sk over over time
            cw_new = (self.disp*np.matmul(self.lap, cw) + dif_bound \
                -self.v * np.matmul(self.fd, cw) - inhomog)/div
            
            sk_new = np.zeros(self.x_steps)
            sk_new[self.x_start:self.x_stop] = self.a_k * \
                ((1-self.f)* self.sorpt_isotherm(cw_soil)-sk_soil)*(self.rho_s/self.n_e)

        else:
            inhomog = self.a_k*((1-self.f)*self.sorpt_isotherm(cw)-sk)*(self.rho_s/self.n_e)
            div = (self.f*self.sorpt_derivat(cw)*(self.rho_s/self.n_e))+1

            cw_new = (self.disp*np.matmul(self.lap, cw) + dif_bound \
            - self.v*np.matmul(self.fd, cw) - inhomog)/div
            sk_new = self.a_k*((1-self.f)*self.sorpt_isotherm(cw)-sk)
        
            
        conc_cw_sk_new = np.ndarray(self.x_steps*2)
        conc_cw_sk_new[:self.x_steps] = cw_new
        conc_cw_sk_new[self.x_steps:] = sk_new

        return conc_cw_sk_new



    # WARNING METHODS ARE NOT USED SO FAR!
    def __solve_ivp_heun(self, u):
        """Explicit heun to integrate ode. WARNING does not work with time, de-
        pendent velocities"""
        for i in range(len(self.t)-1):
            f_0 = self.ad_ode(t=i, conc_cw_ct=u[:,i])
            y_plus_1 = u[:,i]+self.dt*self.ad_ode(t=i, conc_cw_sk=u[:,i])
            f_1 = self.ad_ode(t=i, conc_cw_ct=y_plus_1)

            u[:,i+1] = u[:,i] + self.dt/2*(f_0+f_1)
        return u
    def __arange_velocitys(self) -> np.ndarray:
        """calculates velocity array due to velocity changes during experiment
        Returns:
            np.ndarray: Array with velocities per time step
        """
        sum_dt = 0
        pointer = 0
        # array with velocities per time step
        v_exp = np.zeros(len(self.t))
        for index, elem in enumerate(v_exp):
            if sum_dt >= self.meas_day[-1]:
                break
            else:
                sum_dt = sum_dt + self.dt
                if sum_dt < self.meas_day[pointer]:
                    v_exp[index] = self.v[pointer]
                else:
                    v_exp[index] = self.v[pointer]
                    pointer += 1
        
        # Warning: fill velocities from last measpoint to end with last velocities
        #v_exp[np.where(v_exp == 0)[0][0]:-1] = self.v[-1]

        return v_exp