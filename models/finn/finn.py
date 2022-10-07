"""
Finite Volume Neural Network implementation with PyTorch.
"""

from decimal import DivisionByZero
from types import DynamicClassAttribute
import torch.nn as nn
import torch as th
from torchdiffeq import odeint
import time
import numpy as np
#from torchdiffeq import odeint_adjoint as odeint


import time


class FINN(nn.Module):
    """
    This is the parent FINN class. This class initializes all sharable parameters
    between different implementations to be inherited to each of the implementation.
    It also contains the initialization of the function_learner and reaction_learner
    NN which learns the constitutive relationships (or the flux multiplier) and
    reaction functions, respectively.
    """
    
    def __init__(self, u, D, BC, layer_sizes, device, mode,
                 config, learn_coeff, learn_stencil, bias, sigmoid):
        """
        Constructor.
        
        Inputs:            
        :param u: the unknown variable
        :type u: th.tensor[len(t), Nx, Ny, num_vars]
        
        :param D: diffusion coefficient
        :type D: np.array[num_vars] --- th.tensor is also accepted
        
        :param BC: the boundary condition values. In case of Dirichlet BC, this
        contains the scalar values. In case of Neumann, this contains the flux
        values.
        :type BC: np.array[num_bound, num_vars] --- th.tensor is also accepted
        
        :param layer_sizes: a list of the hidden nodes for each layer (including
        the input and output features)
        :type layer_sizes: list[num_hidden_layers + 2]
        
        :param device: the device to perform simulation
        :type device: th.device
        
        :param mode: mode of simulation ("train" or "test")
        :type mode: str
        
        :param config: configuration of simulation parameters
        :type config: dict
        
        :param learn_coeff: a switch to set diffusion coefficient to be learnable
        :type learn_coeff: bool
        
        :param learn_stencil: a switch to set the numerical stencil to be learnable
        :type learn_stencil: bool
        
        :param bias: a bool value to determine whether to use bias values in the
        function_learner
        :type bias bool
        
        :param sigmoid: a bool value to determine whether to use sigmoid at the
        output layer
        :type sigmoid: bool
        
        Output:
        :return: the full field solution of u from time t0 until t_end
        :rtype: th.tensor[len(t), Nx, Ny, num_vars]

        """
        
        super(FINN, self).__init__()
        
        self.device = device
        self.Nx = u.size()[1]
        self.BC = th.tensor(BC, dtype=th.double).to(device=self.device)
        self.layer_sizes = layer_sizes
        self.mode = mode
        self.cfg = config
        self.bias = bias
        self.sigmoid = sigmoid
        
        
        if not learn_coeff:
            self.D = th.tensor(D, dtype=th.double, device=self.device)
        else:
            # #sets as learnable parameter
            self.D = nn.Parameter(th.tensor(D, dtype=th.double,
                                            device=self.device))
        
        if not learn_stencil:
            self.stencil = th.tensor([-1.0, 1.0], dtype=th.double,
                                     device=self.device)
        else:
            self.stencil = th.tensor(
                # # learnable stencil, mean=-1.0, std=0.1
                [th.normal(th.tensor([-1.0]), th.tensor([0.1])),
                 th.normal(th.tensor([1.0]), th.tensor([0.1]))],
                dtype=th.double, device=self.device)
            self.stencil = nn.Parameter(self.stencil)
    
    def function_learner(self):
        """
        This function constructs a feedforward NN required for calculation
        of constitutive function (or flux multiplier) as a function of u.
        """
        layers = list()
        
        for layer_idx in range(len(self.layer_sizes) - 1):
            layer = nn.Linear(
                in_features=self.layer_sizes[layer_idx],
                out_features=self.layer_sizes[layer_idx + 1],
                bias=self.bias
                ).to(device=self.device)
            layers.append(layer)
        
            if layer_idx < len(self.layer_sizes) - 2 or not self.sigmoid:
                layers.append(nn.Tanh())
            else:
                # Use sigmoid function to keep the values strictly positive
                # (all outputs have the same sign)
                layers.append(nn.Sigmoid())
        return nn.Sequential(*nn.ModuleList(layers))
    

class Wrapper(nn.Module):
    def __init__(self, f):

        super().__init__()
        self.f = f

    def forward(self, t, u):
        return self.f(t, u)



class FINN_Burger(FINN):
    """
    This is the inherited FINN class for the Burger equation implementation.
    This class inherits all parameter from the parent FINN class.
    """
    
    def __init__(self, u, D, BC, dx, layer_sizes, device, mode="train",
                 config=None, learn_coeff=False, learn_stencil=False,
                 bias=False, sigmoid=False):
        """
        Constructor.
        
        Inputs:
        Same with the parent FINN class, with the addition of dx (the spatial resolution)
        """
        
        super().__init__(u, D, BC, layer_sizes, device, mode, config, learn_coeff,
                         learn_stencil, bias, sigmoid)
        
        self.dx = dx
        
        # Initialize the function_learner to learn the first order flux multiplier
        self.func_nn = self.function_learner().to(device=self.device)

        self.wrapper = Wrapper(self.state_kernel)
    
    """
    TODO: Implement flux kernel for test (if different BC is used)
    """
        
    def flux_kernel(self, t, u):
        """
        This function defines the flux kernel for training, which takes ui and its
        neighbors as inputs, and returns the integrated flux approximation (up to
        second order derivatives)
        """
        
        # Approximate the first order flux multiplier
        a = self.func_nn(u.unsqueeze(-1))
        
        # Apply the ReLU function for upwind scheme to prevent numerical
        # instability
        a_plus = th.relu(a[...,0])
        a_min = -th.relu(-a[...,0])
        
        
        ## Calculate fluxes at the left boundary of control volumes i
        
        # Calculate the flux at the left domain boundary
        left_bound_flux = (self.D*(self.stencil[0]*u[0] +
                            self.stencil[1]*self.BC[0,0]) -\
                            a_plus[0]/self.dx*(-self.stencil[0]*u[0] -
                            self.stencil[1]*self.BC[0,0]))
        
        #print(a_plus[1:])
        # Calculate the fluxes between control volumes i and their left neighbors
        left_neighbors = self.D*(self.stencil[0]*u[1:] +
                            self.stencil[1]*u[:-1]) -\
                            a_plus[1:]/self.dx*(-self.stencil[0]*u[1:] -
                            self.stencil[1]*u[:-1])
                            
        
        # Concatenate the left fluxes
        left_flux = th.cat((left_bound_flux, left_neighbors))
        
        ## Calculate fluxes at the right boundary of control volumes i
        
        # Calculate the flux at the right domain boundary
        right_bound_flux = (self.D*(self.stencil[0]*u[-1] +
                            self.stencil[1]*self.BC[1,0]) -\
                            a_min[-1]/self.dx*(self.stencil[0]*u[-1] +
                            self.stencil[1]*self.BC[1,0]))
                 
        # Calculate the fluxes between control volumes i and their right neighbors
        right_neighbors = self.D*(self.stencil[0]*u[:-1] +
                            self.stencil[1]*u[1:]) -\
                            a_min[:-1]/self.dx*(self.stencil[0]*u[:-1] +
                            self.stencil[1]*u[1:])
        
        # Concatenate the right fluxes
        right_flux = th.cat((right_neighbors, right_bound_flux))
        
        
        # Integrate the fluxes at all boundaries of control volumes i
        flux = left_flux + right_flux
        
        return flux
    
    def state_kernel(self, t, u):
        """
        This function defines the state kernel for training, which takes the
        fluxes as inputs, and returns du/dt)
        """
        
        #t = t.to(device="cuda")
        #u = u.to(device="cuda")
        flux = self.flux_kernel(t, u)
        
        # Since there is no reaction term to be learned, du/dt = fluxes
        state = flux
        
        return state#.to(device="cpu")
    
    def forward(self, t, u):
        """
        This function integrates du/dt through time using the Neural ODE method
        """
        
        # The odeint function receives the function state_kernel that calculates
        # du/dt, the initial condition u[0], and the time at which the values of
        # u will be saved t
        #pred = odeint(self.state_kernel, u[0].to(device="cpu"), t.to(device="cpu"))
        pred = odeint(self.state_kernel, u[0], t)
        #pred = odeint(self.wrapper, u[0], t)
       
        return pred#.to(device="cuda")
    
class FINN_DiffSorp(FINN):
    """
    This is the inherited FINN class for the diffusion-sorption equation implementation.
    This class inherits all parameter from the parent FINN class.
    """
    def __init__(self, u, D, BC, dx, layer_sizes, device, mode="train",
                 config=None, learn_coeff=False, learn_stencil=False, bias=True,
                 sigmoid=True):
        
        super().__init__(u, D, BC, layer_sizes, device, mode, config, learn_coeff,
                         learn_stencil, bias, sigmoid)
        
        """
        Constructor.
        
        Inputs:
        Same with the parent FINN class, with the addition of dx (the spatial resolution)
        """
        
        self.dx = dx
        
        # Initialize the function_learner to learn the retardation factor function
        self.func_nn = self.function_learner().to(device=self.device)
        # Initialize the multiplier of the retardation factor function (denormalization)
        self.p_exp = nn.Parameter(th.tensor([10],dtype=th.double))
        
    
    """
    TODO: Implement flux kernel for test (if different BC is used)
    """
        
    def flux_kernel(self, t, u):
        """
        This function defines the flux kernel for training, which takes ui and its
        neighbors as inputs, and returns the integrated flux approximation (up to
        second order derivatives)
        """
        
        # Separate u into c and ct
        c = u[...,0]
        ct = u[...,1]
        # Approximate 1/retardation_factor        
        ret = (self.func_nn(c.unsqueeze(-1)) * 10**self.p_exp)[...,0]

        # ret = (self.func_nn(c.unsqueeze(-1)))[...,0]
        
        ## Calculate fluxes at the left boundary of control volumes i
        
        ## For c
        # Calculate the flux at the left domain boundary
        left_bound_flux_c = (self.D[0]*ret[0]*(self.stencil[0]*c[0] +
                            self.stencil[1]*self.BC[0,0])).unsqueeze(0)
        
        
        # Calculate the fluxes between control volumes i and their left neighbors
        left_neighbors_c = self.D[0]*ret[1:]*(self.stencil[0]*c[1:] +
                            self.stencil[1]*c[:-1])
        
        # Concatenate the left fluxes
        left_flux_c = th.cat((left_bound_flux_c, left_neighbors_c))
        
        ## For ct
        # Calculate the flux at the left domain boundary
        left_bound_flux_ct = (self.D[1]*(self.stencil[0]*c[0] +
                            self.stencil[1]*self.BC[0,1])).unsqueeze(0)
              
        # Calculate the fluxes between control volumes i and their left neighbors              
        left_neighbors_ct = self.D[1]*(self.stencil[0]*c[1:] +
                            self.stencil[1]*c[:-1])
        
        # Concatenate the left fluxes
        left_flux_ct = th.cat((left_bound_flux_ct, left_neighbors_ct))
        
        # Stack the left fluxes of c and ct together
        left_flux = th.stack((left_flux_c, left_flux_ct), dim=len(c.size()))
        
        
        ## Calculate fluxes at the right boundary of control volumes i
        
        ## For c
        # Calculate the Cauchy condition for the right domain boundary
        right_BC = self.D[0]*self.dx*(c[-2]-c[-1])
        
        # Calculate the flux at the right domain boundary     
        right_bound_flux_c = (self.D[0]*ret[-1]*(self.stencil[0]*c[-1] +
                            self.stencil[1]*right_BC)).unsqueeze(0)
        
        # Calculate the fluxes between control volumes i and their right neighbors 
        right_neighbors_c = self.D[0]*ret[:-1]*(self.stencil[0]*c[:-1] +
                            self.stencil[1]*c[1:])
                        
        # Concatenate the right fluxes
        right_flux_c = th.cat((right_neighbors_c, right_bound_flux_c))
        
        ## For ct
        # Calculate the flux at the right domain boundary 
        right_bound_flux_ct = (self.D[1]*(self.stencil[0]*c[-1] +
                            self.stencil[1]*right_BC)).unsqueeze(0)
        
        # Calculate the fluxes between control volumes i and their right neighbors
        right_neighbors_ct = self.D[1]*(self.stencil[0]*c[:-1] +
                            self.stencil[1]*c[1:])
        
        # Concatenate the right fluxes
        right_flux_ct = th.cat((right_neighbors_ct, right_bound_flux_ct))
        
        # Stack the right fluxes of c and ct together
        right_flux = th.stack((right_flux_c, right_flux_ct), dim=len(c.size()))
        
        
        # Integrate the fluxes at all boundaries of control volumes i
        flux = left_flux + right_flux
        
        return flux
    
    def state_kernel(self, t, u):
        """
        This function defines the state kernel for training, which takes the
        fluxes as inputs, and returns du/dt)
        """
        
        flux = self.flux_kernel(t, u)
        
        # Since there is no reaction term to be learned, du/dt = fluxes
        state = flux
        return state
    
    def forward(self, t, u):
        """
        This function integrates du/dt through time using the Neural ODE method
        """
        
        # The odeint function receives the function state_kernel that calculates
        # du/dt, the initial condition u[0], and the time at which the values of
        # u will be saved t
        pred = odeint(self.state_kernel, u[0], t)
        
        return pred


class FINN_DiffAD2ss(FINN):
    """
    This is the inherited FINN class for the diffusion-sorption equation implementation.
    This class inherits all parameter from the parent FINN class.
    """
    def __init__(self, u, D, BC, dx, layer_sizes, device, rho_s, f, k_d, beta, n_e, 
                alpha, v_e, t_steps, learn_f_hyd, learn_r_hyd, learn_g_hyd, sand, 
                D_sand=None,
                n_e_sand=None, x_start_soil=None, x_stop_soil=None, x_steps_soil=None,
                x_steps_sand=None, alpha_l_sand=None, v_e_sand=None, mode="train",
                config=None, learn_coeff=True, learn_stencil=False, bias=True,
                sigmoid=True, learn_f=False, learn_sk=True, learn_k_d=True, learn_ve=True):
        
        super().__init__(u, D, BC, layer_sizes, device, mode, config, learn_coeff,
                         learn_stencil, bias, sigmoid)
        
        """
        Constructor.
        
        Inputs:
        Same with the parent FINN class, with the addition of dx (the spatial resolution)
        
        :param rho_s: soil density
        :type rho_s: np.array

        :param f: share of instantaneously sorbed concentration

        :k_d: freundlich isotherm parameter

        :beta: freundlich isotherm proportional coefficient
        
        :n_e: effective porosity

        :alpha: first order rate constant

        :v_e: effective velocity 
        """
        # potentially learnable parameters
        if not learn_f:
            self.f = th.tensor(f, dtype=th.double, device=self.device)
        else:
            self.f = nn.Parameter(th.tensor(f, dtype=th.double,
                                            device=self.device))
        if not learn_k_d:
            self.k_d = th.tensor(k_d,dtype=th.double, device=self.device)
        else:
            self.k_d = nn.Parameter(th.tensor(k_d, dtype=th.double, device=self.device))

        if not learn_ve:
            self.v_e = th.tensor(v_e, dtype=th.double, device=self.device)
        else:
            self.v_e = nn.Parameter(th.tensor(v_e, dtype=th.double, device=self.device))

        if learn_sk:
            self.func_sk = self.function_learner().to(device=self.device)
            self.sk_fac = nn.Parameter(th.tensor([1.0], dtype=th.double))
        if learn_f_hyd:
            self.func_f = self.function_learner().to(device=self.device)
            self.f_fac = nn.Parameter(th.tensor([1.0], dtype=th.double))
        if learn_r_hyd:
            self.func_r = self.function_learner().to(device=self.device)
            self.ret_fac = nn.Parameter(th.tensor([1.0], dtype=th.double))

        if learn_g_hyd:
            self.func_g = self.function_learner().to(device=self.device)
            self.g_fac = nn.Parameter(th.tensor([1.0], dtype=th.double))
        
        
        # For testing FV solver without optimization
        #self.z = nn.Parameter(th.tensor([0],dtype=th.double, requires_grad=True))
        # potentially non-learnable parameters
        self.dx = th.tensor(dx, dtype=th.double, device=self.device)
        self.rho_s = th.tensor(rho_s, dtype=th.double, device=self.device)        
        self.beta = th.tensor(beta, dtype=th.double, device=self.device)
        self.n_e  = th.tensor(n_e, dtype=th.double, device=self.device)
        self.alpha = th.tensor(alpha, dtype = th.double, device=self.device)
        self.learn_sk = learn_sk
        self.learn_f_hyd = learn_f_hyd
        self.learn_g_hyd = learn_g_hyd
        self.learn_r_hyd = learn_r_hyd
        self.sand = sand
        if self.sand:
            self.D_sand = th.tensor(D_sand, dtype=th.double, device=self.device)
            self.n_e_sand = th.tensor(n_e_sand, dtype=th.double, device=self.device)
            self.x_start = th.tensor(x_start_soil, dtype=th.int, device=self.device)
            self.x_stop = th.tensor(x_stop_soil, dtype=th.int, device=self.device)
            self.x_steps_soil = th.tensor(x_steps_soil, dtype=th.int, device=self.device)
            self.x_steps_sand = th.tensor(x_steps_sand, dtype=th.int, device=self.device)
            self.alpha_l_sand = th.tensor(alpha_l_sand, dtype=th.double, device=self.device)
            self.v_e_sand = th.tensor(v_e_sand, dtype=th.double, device=self.device)
        
        # counter for time dependent neural network
        # which of the timestep input feature should be used. The others are 
        # set to zero
        self.t_steps = t_steps

        
    
    """
    TODO: Implement flux kernel for test (if different BC is used)
    """
        
    def flux_kernel(self, t, u):
        """
        This function defines the flux kernel for training, which takes ui and its
        neighbors as inputs, and returns the integrated flux approximation (up to
        second order derivatives)
        ## u = [[c, sk]] all spatials at one time step
        """
        # Separate u into c and ct
        c = u[...,0]
        sk = u[...,1]

        if self.sand:
            
            cw_soil= c[self.x_start:self.x_stop]
            sk_soil = sk[self.x_start:self.x_stop]

            v_soil = self.v_e
            v_soil_plus = th.relu(v_soil)
            v_soil_min = -th.relu(-v_soil)

            v_sand = self.v_e_sand
            v_sand_plus = th.relu(v_sand)
            v_sand_min = -th.relu(-v_sand)
            

            # top boundary fluxes
            top_bound_flux = (self.D_sand/(self.dx**2)*(self.stencil[0]*c[0] +
                            self.stencil[1]*self.BC[0]) -\
                            v_sand_plus/self.dx*(-self.stencil[0]*c[0] -
                            self.stencil[1]*self.BC[0])).unsqueeze(0)

            # start: 7, stop: 48, X_STEPS: 56
            # c[:self.x_start-1] = c1 ... c6
            # c[1:self.x_start] = c2 ... c7
            # c[self.x_start:self.x_stop] = c8 ... c48
            # c[self.x_stop+1:] = c50 ... c56
            # c[self.x_stop:-1] = c49 ... c55
            # c[self.x_stop:]) = c49 ... c56
            top_flux_sand_top = self.D_sand/(self.dx**2)*(self.stencil[0]*c[1:self.x_start] +
                            self.stencil[1]*c[:self.x_start-1]) -\
                            v_sand_plus/self.dx*(-self.stencil[0]*c[1:self.x_start] -
                            self.stencil[1]*c[:self.x_start-1])
            top_flux_soil = self.D/(self.dx**2)*(self.stencil[0]*c[self.x_start:self.x_stop] +
                            self.stencil[1]*c[self.x_start-1:self.x_stop-1]) -\
                            v_soil_plus/self.dx*(-self.stencil[0]*c[self.x_start:self.x_stop] -
                            self.stencil[1]*c[self.x_start-1:self.x_stop-1])
            top_flux_sand_bot = self.D_sand/self.dx**2 * (self.stencil[0]*c[self.x_stop:]+self.stencil[1]*c[self.x_stop-1:-1]) - \
                v_sand_plus/self.dx * (-self.stencil[0]*c[self.x_stop:]-self.stencil[1]*c[self.x_stop-1:-1])
            
            top_flux = th.cat((top_bound_flux, top_flux_sand_top, top_flux_soil, top_flux_sand_bot))

            # bot boundary fluxes
            bot_flux_sand_top = self.D_sand/self.dx**2 *(self.stencil[0]*c[:self.x_start]+self.stencil[1]*c[1:self.x_start+1]) - \
                v_sand_min/self.dx * (self.stencil[0]*c[:self.x_start]+self.stencil[1]*c[1:self.x_start+1])
            bot_flux_soil = self.D/self.dx**2 *(self.stencil[0]*c[self.x_start:self.x_stop] + self.stencil[1]*c[self.x_start+1:self.x_stop+1]) - \
                v_soil_min/self.dx * (self.stencil[0]*c[self.x_start:self.x_stop] + self.stencil[1]*c[self.x_start+1:self.x_stop+1])
            bot_flux_sand_bot = self.D_sand/self.dx**2 * (self.stencil[0]*c[self.x_stop:-1]+self.stencil[1]*c[self.x_stop+1:]) - \
                v_sand_min/self.dx * (self.stencil[0]*c[self.x_stop:-1]+self.stencil[1]*c[self.x_stop+1:])
            bot_bound_flux = th.tensor(0).unsqueeze(0)

            bot_flux = th.cat((bot_flux_sand_top, bot_flux_soil, bot_flux_sand_bot, bot_bound_flux))
    
            if not self.learn_sk:
                sk  = sk
                
            else:    
                time_vec = th.ones([self.Nx])*t
                t_c = th.stack((c.float(),time_vec), dim=1)
                sk = self.func_sk(t_c)*th.abs(self.sk_fac)
                sk = sk.squeeze(-1)
                
            if not self.learn_f_hyd:
                f_hyd = th.zeros(self.Nx)
                f_hyd[self.x_start:self.x_stop] = -self.alpha*self.rho_s/self.n_e*(1-self.f)*(self.k_d*cw_soil**(self.beta-1))
                
            else:
                time_vec = th.ones([self.Nx])*t
                t_c = th.stack((c.float(),time_vec), dim=1)
                f_hyd = self.func_f(t_c)*self.f_fac
                f_hyd = f_hyd.squeeze(-1)
            
            if not self.learn_r_hyd:
                ret = th.ones(self.Nx)
                ret[self.x_start:self.x_stop] = (self.f*(self.k_d*self.beta*cw_soil**(self.beta-1))*(self.rho_s/self.n_e))+1

            else:
                time_vec = th.ones([self.Nx])*t
                t_c = th.stack((c.float(), time_vec), dim=1)

                # phsics informed: ret > 1 and larger than sigmoid output ([0,1])
                ret = 1+self.func_r(t_c)*(10**self.ret_fac)
                ret = ret.squeeze(-1)
            
            if not self.learn_g_hyd:
                g_hyd = th.zeros(self.Nx)
                g_hyd[self.x_start:self.x_stop] = self.alpha*self.rho_s/self.n_e*sk_soil
                
            else:
                time_vec = th.ones([self.Nx])*t
                t_c = th.stack((c.float(), time_vec), dim=1)
                g_hyd = self.func_g(t_c)*th.abs(self.g_fac)
                g_hyd = g_hyd.squeeze(-1)
            
            # Integrate the fluxes at all boundaries of control volumes i
            flux_c = (top_flux + bot_flux+f_hyd*c+g_hyd)/ret
            th.set_printoptions(precision=10)
            

            # total concentration
            flux_sk=th.zeros(self.Nx)
            flux_sk[self.x_start:self.x_stop] = -f_hyd[self.x_start:self.x_stop]\
                *(self.n_e/self.rho_s)*cw_soil-g_hyd[self.x_start:self.x_stop]\
                    *(self.n_e/self.rho_s)
            #print(flux_sk)
            #time.sleep(2)
            flux = th.stack((flux_c, flux_sk), dim=len(c.size()))
            return flux

        else:
            a = th.ones([self.Nx,1], dtype=th.double, device=self.device)*self.v_e
            # Apply the ReLU function for upwind scheme to prevent numerical
            # instability
            a_plus = th.relu(a[...,0])
            
            # a_min = -0 since a = 30
            a_min = -th.relu(-a[...,0])
            
            ## Calculate fluxes at the top boundary of control volumes i

            # Calculate the flux at the top domain boundary
            top_bound_flux = (self.D/(self.dx**2)*(self.stencil[0]*c[0] +
                                self.stencil[1]*self.BC[0]) -\
                                a_plus[0]/self.dx*(-self.stencil[0]*c[0] -
                                self.stencil[1]*self.BC[0])).unsqueeze(0)
                    
            # Calculate the fluxes between control volumes i and their top neighbors
            top_neighbors = self.D/(self.dx**2)*(self.stencil[0]*c[1:] +
                                self.stencil[1]*c[:-1]) -\
                                a_plus[1:]/self.dx*(-self.stencil[0]*c[1:] -
                                self.stencil[1]*c[:-1])
            
            # Concatenate the left fluxes
            top_flux = th.cat((top_bound_flux, top_neighbors))
            ## Calculate fluxes at the right boundary of control volumes i
            
            # Calculate the flux at the right domain boundary -> Neumann -> zero??
            bot_bound_flux = th.tensor(0).unsqueeze(0)
            
            # Calculate the fluxes between control volumes i and their right neighbors
            bot_neighbors = self.D/(self.dx**2)*(self.stencil[0]*c[:-1] +
                                self.stencil[1]*c[1:]) -\
                                a_min[:-1]/self.dx*(self.stencil[0]*c[:-1] + \
                                self.stencil[1]*c[1:])
            
            # Concatenate the right fluxes
            bot_flux = th.cat((bot_neighbors, bot_bound_flux))
            
            if not self.learn_sk:
                sk  = sk
                
            else:    
                time_vec = th.ones([self.Nx])*t
                t_c = th.stack((c.float(),time_vec), dim=1)
                sk = self.func_sk(t_c)*th.abs(self.sk_fac)
                sk = sk.squeeze(-1)
                
            if not self.learn_f_hyd:
                f_hyd = -self.alpha*self.rho_s/self.n_e*(1-self.f)*(self.k_d*c**(self.beta-1))
                
            else:
                time_vec = th.ones([self.Nx])*t
                t_c = th.stack((c.float(),time_vec), dim=1)
                f_hyd = self.func_f(t_c)*self.f_fac
                f_hyd = f_hyd.squeeze(-1)
            
            if not self.learn_r_hyd:
                ret = (self.f*(self.k_d*self.beta*c**(self.beta-1))*(self.rho_s/self.n_e))+1

            else:
                time_vec = th.ones([self.Nx])*t
                t_c = th.stack((c.float(), time_vec), dim=1)

                # phsics informed: ret > 1 and larger than sigmoid output ([0,1])
                ret = 1+self.func_r(t_c)*(10**self.ret_fac)
                ret = ret.squeeze(-1)
            
            if not self.learn_g_hyd:
                g_hyd = self.alpha*self.rho_s/self.n_e*sk
                
            else:
                time_vec = th.ones([self.Nx])*t
                t_c = th.stack((c.float(), time_vec), dim=1)
                g_hyd = self.func_g(t_c)*th.abs(self.g_fac)
                g_hyd = g_hyd.squeeze(-1)
                
            # Integrate the fluxes at all boundaries of control volumes i
            flux_c = (top_flux + bot_flux+f_hyd*c+g_hyd)/ret

            # total concentration
            flux_sk = -f_hyd*(self.n_e/self.rho_s)*c-g_hyd*(self.n_e/self.rho_s)
            flux = th.stack((flux_c, flux_sk), dim=len(c.size()))
            return flux
    
    def state_kernel(self, t, u):
        """
        This function defines the state kernel for training, which takes the
        fluxes as inputs, and returns du/dt)
        """        

        state = self.flux_kernel(t, u)
        return state
    
    def forward(self, t, u):
        """
        This function integrates du/dt through time using the Neural ODE method
        """
        
        # The odeint function receives the function state_kernel that calculates
        # du/dt, the initial condition u[0], and the time at which the values of
        # u will be saved t

        pred = odeint(self.state_kernel, u[0], t, method="euler")
        return pred

class FINN_DiffReact(FINN):
    """
    This is the inherited FINN class for the diffusion-reaction equation implementation.
    This class inherits all parameter from the parent FINN class.
    """
    def __init__(self, u, D, BC, dx, dy, layer_sizes, device, mode="train",
                 config=None, learn_coeff=False, learn_stencil=False, bias=False,
                 sigmoid=False):
        
        super().__init__(u, D, BC, layer_sizes, device, mode, config, learn_coeff,
                         learn_stencil, bias, sigmoid)
        
        """
        Constructor.
        
        Inputs:
        Same with the parent FINN class, with the addition of dx and dy (the
        spatial resolution)
        """
        
        self.dx = dx
        self.dy = dy
        
        self.Ny = u.size()[2]
        
        # Initialize the reaction_learner to learn the reaction term
        self.func_nn = self.function_learner().to(device=self.device)

        self.right_flux = th.zeros(49, 49, 2, device=self.device)
        self.left_flux = th.zeros(49, 49, 2, device=self.device)

        self.bottom_flux = th.zeros(49, 49, 2, device=self.device)
        self.top_flux = th.zeros(49, 49, 2, device=self.device)
        
    
    """
    TODO: Implement flux kernel for test (if different BC is used)
    """
        
    def flux_kernel(self, t, u):
        """
        This function defines the flux kernel for training, which takes ui and its
        neighbors as inputs, and returns the integrated flux approximation (up to
        second order derivatives)
        """
        
        # Separate u into act and inh
        act = u[...,0]
        inh = u[...,1]
        
        ## Calculate fluxes at the left boundary of control volumes i
        
        ## For act
        # Calculate the flux at the left domain boundary
        left_bound_flux_act = th.cat(self.Ny * [self.BC[0,0].unsqueeze(0)]).unsqueeze(0)
                            
        # Calculate the fluxes between control volumes i and their left neighbors 
        left_neighbors_act = self.D[0]*(self.stencil[0]*act[1:] +
                            self.stencil[1]*act[:-1])

        # Concatenate the left fluxes
        left_flux_act = th.cat((left_bound_flux_act, left_neighbors_act))
                
        ## For inh
        # Calculate the flux at the left domain boundary
        left_bound_flux_inh = th.cat(self.Ny * [self.BC[0,1].unsqueeze(0)]).unsqueeze(0)
                
        # Calculate the fluxes between control volumes i and their left neighbors
        left_neighbors_inh = self.D[1]*(self.stencil[0]*inh[1:] +
                            self.stencil[1]*inh[:-1])

        # Concatenate the left fluxes
        left_flux_inh = th.cat((left_bound_flux_inh, left_neighbors_inh))

        # Stack the left fluxes of act and inh together
        left_flux = th.stack((left_flux_act, left_flux_inh), dim=len(act.size()))

        ## Calculate fluxes at the right boundary of control volumes i
        
        ## For act
        # Calculate the flux at the right domain boundary
        right_bound_flux_act = th.cat(self.Ny * [self.BC[1,0].unsqueeze(0)]).unsqueeze(0)
                            
        # Calculate the fluxes between control volumes i and their right neighbors 
        right_neighbors_act = self.D[0]*(self.stencil[0]*act[:-1] +
                            self.stencil[1]*act[1:])
        
        # Concatenate the right fluxes
        right_flux_act = th.cat((right_neighbors_act, right_bound_flux_act))
        
        ## For inh
        # Calculate the flux at the right domain boundary  
        right_bound_flux_inh = th.cat(self.Ny * [self.BC[1,1].unsqueeze(0)]).unsqueeze(0)
           
        # Calculate the fluxes between control volumes i and their right neighbors  
        right_neighbors_inh = self.D[1]*(self.stencil[0]*inh[:-1] +
                            self.stencil[1]*inh[1:])

        # Concatenate the right fluxes
        right_flux_inh = th.cat((right_neighbors_inh, right_bound_flux_inh))
        
        # Stack the right fluxes of act and inh together
        right_flux = th.stack((right_flux_act, right_flux_inh), dim=len(act.size()))
        
        ## Calculate fluxes at the bottom boundary of control volumes i
        
        ## For act
        # Calculate the flux at the bottom domain boundary
        bottom_bound_flux_act = th.cat(self.Nx * [self.BC[2,0].unsqueeze(0)]).unsqueeze(-1)
           
        # Calculate the fluxes between control volumes i and their bottom neighbors                   
        bottom_neighbors_act = self.D[0]*(self.stencil[0]*act[:,1:] +
                            self.stencil[1]*act[:,:-1])
        
        # Concatenate the bottom fluxes
        bottom_flux_act = th.cat((bottom_bound_flux_act, bottom_neighbors_act), dim=1)
        
        ## For inh
        # Calculate the flux at the bottom domain boundary
        bottom_bound_flux_inh = th.cat(self.Nx * [self.BC[2,1].unsqueeze(0)]).unsqueeze(-1)
                            
        # Calculate the fluxes between control volumes i and their bottom neighbors
        bottom_neighbors_inh = self.D[1]*(self.stencil[0]*inh[:,1:] +
                            self.stencil[1]*inh[:,:-1])
        
        # Concatenate the bottom fluxes
        bottom_flux_inh = th.cat((bottom_bound_flux_inh, bottom_neighbors_inh),dim=1)

        # Stack the bottom fluxes of act and inh together
        bottom_flux = th.stack((bottom_flux_act, bottom_flux_inh), dim=len(act.size()))

        ## Calculate fluxes at the bottom boundary of control volumes i
        
        ## For act
        # Calculate the flux at the top domain boundary
        top_bound_flux_act = th.cat(self.Nx * [self.BC[3,0].unsqueeze(0)]).unsqueeze(-1)
                            
        # Calculate the fluxes between control volumes i and their top neighbors
        top_neighbors_act = self.D[0]*(self.stencil[0]*act[:,:-1] +
                            self.stencil[1]*act[:,1:])
        
        # Concatenate the top fluxes
        top_flux_act = th.cat((top_neighbors_act, top_bound_flux_act), dim=1)
        
        ## For inh
        # Calculate the flux at the top domain boundary
        top_bound_flux_inh = th.cat(self.Nx * [self.BC[3,1].unsqueeze(0)]).unsqueeze(-1)
                  
        # Calculate the fluxes between control volumes i and their top neighbors
        top_neighbors_inh = self.D[1]*(self.stencil[0]*inh[:,:-1] +
                            self.stencil[1]*inh[:,1:])
        
        # Concatenate the top fluxes
        top_flux_inh = th.cat((top_neighbors_inh, top_bound_flux_inh), dim=1)
        
        # Stack the top fluxes of act and inh together
        top_flux = th.stack((top_flux_act, top_flux_inh), dim=len(act.size()))
        
        # Integrate the fluxes at all boundaries of control volumes i
        flux = left_flux + right_flux + bottom_flux + top_flux
        
        return flux
    
    def state_kernel(self, t, u):
        """
        This function defines the state kernel for training, which takes the
        fluxes as inputs, and returns du/dt)
        """
        
        flux = self.flux_kernel(t, u)
        
        # Add the reaction term to the fluxes term to obtain du/dt
        state = flux + self.func_nn(u)
        
        return state
    
    def forward(self, t, u):
        """
        This function integrates du/dt through time using the Neural ODE method
        """
        
        # The odeint function receives the function state_kernel that calculates
        # du/dt, the initial condition u[0], and the time at which the values of
        # u will be saved t
        pred = odeint(self.state_kernel, u[0], t)
        
        return pred
    

class FINN_AllenCahn(FINN):
    """
    This is the inherited FINN class for the Burger equation implementation.
    This class inherits all parameter from the parent FINN class.
    """
    
    def __init__(self, u, D, BC, dx, layer_sizes, device, mode="train",
                 config=None, learn_coeff=False, learn_stencil=False,
                 bias=False, sigmoid=False):
        """
        Constructor.
        
        Inputs:
        Same with the parent FINN class, with the addition of dx (the spatial resolution)
        """
        
        super().__init__(u, D, BC, layer_sizes, device, mode, config, learn_coeff,
                         learn_stencil, bias, sigmoid)
        
        self.dx = dx
        
        # Initialize the function_learner to learn the first order flux multiplier
        self.func_nn = self.function_learner().to(device=self.device)
        
        # Initialize the multiplier of the retardation factor function (denormalization)
        self.p_mult = nn.Parameter(th.tensor([10.0],dtype=th.double))
    
    """
    TODO: Implement flux kernel for test (if different BC is used)
    """
        
    def flux_kernel(self, t, u):
        """
        This function defines the flux kernel for training, which takes ui and its
        neighbors as inputs, and returns the integrated flux approximation (up to
        second order derivatives)
        """
        
        ## Calculate fluxes at the left boundary of control volumes i
        
        # Calculate the flux at the left domain boundary
        left_bound_flux = self.D/10*(self.stencil[0]*u[0] +
                            self.stencil[1]*u[-1])
                            
        # Calculate the fluxes between control volumes i and their left neighbors
        left_neighbors = self.D/10*(self.stencil[0]*u[1:] +
                            self.stencil[1]*u[:-1])
        
        # Concatenate the left fluxes
        left_flux = th.cat((left_bound_flux, left_neighbors))
        
        ## Calculate fluxes at the right boundary of control volumes i
        
        # Calculate the flux at the right domain boundary
        right_bound_flux = self.D/10*(self.stencil[0]*u[-1] +
                            self.stencil[1]*u[0])
                 
        # Calculate the fluxes between control volumes i and their right neighbors
        right_neighbors = self.D/10*(self.stencil[0]*u[:-1] +
                            self.stencil[1]*u[1:])
        
        # Concatenate the right fluxes
        right_flux = th.cat((right_neighbors, right_bound_flux))
        
        
        # Integrate the fluxes at all boundaries of control volumes i
        flux = left_flux + right_flux
        
        return flux
    
    def state_kernel(self, t, u):
        """
        This function defines the state kernel for training, which takes the
        fluxes as inputs, and returns du/dt)
        """
        
        flux = self.flux_kernel(t, u)
        
        # Since there is no reaction term to be learned, du/dt = fluxes
        state = flux + self.func_nn(u.unsqueeze(-1)).squeeze()*self.p_mult
        
        return state
    
    def forward(self, t, u):
        """
        This function integrates du/dt through time using the Neural ODE method
        """
        
        # The odeint function receives the function state_kernel that calculates
        # du/dt, the initial condition u[0], and the time at which the values of
        # u will be saved t
        pred = odeint(self.state_kernel, u[0], t)
        
        return pred


class FINN_Burger2D(FINN):
    """
    This is the inherited FINN class for the diffusion-reaction equation implementation.
    This class inherits all parameter from the parent FINN class.
    """
    def __init__(self, u, D, BC, dx, dy, layer_sizes, device, mode="train",
                 config=None, learn_coeff=False, learn_stencil=False, bias=False,
                 sigmoid=False):
        
        super().__init__(u, D, BC, layer_sizes, device, mode, config, learn_coeff,
                         learn_stencil, bias, sigmoid)
        
        """
        Constructor.
        
        Inputs:
        Same with the parent FINN class, with the addition of dx and dy (the
        spatial resolution)
        """
        
        self.dx = dx
        self.dy = dy
        
        self.Ny = u.size()[2]
        
        # Initialize the reaction_learner to learn the reaction term
        self.func_nn = self.function_learner().to(device=self.device)

        self.right_flux = th.zeros(49, 49, 1, device=self.device)
        self.left_flux = th.zeros(49, 49, 1, device=self.device)

        self.bottom_flux = th.zeros(49, 49, 1, device=self.device)
        self.top_flux = th.zeros(49, 49, 1, device=self.device)
        
        layers = list()
        
        for layer_idx in range(len(self.layer_sizes) - 1):
            layer = nn.Linear(
                in_features=self.layer_sizes[layer_idx],
                out_features=self.layer_sizes[layer_idx + 1],
                bias=self.bias
                ).to(device=self.device)
            layers.append(layer)
        
            if layer_idx < len(self.layer_sizes) - 2:
                layers.append(nn.Tanh())
        
        self.func_nn = nn.Sequential(*nn.ModuleList(layers))
        
    
    """
    TODO: Implement flux kernel for test (if different BC is used)
    """
        
    def flux_kernel(self, t, u):
        """
        This function defines the flux kernel for training, which takes ui and its
        neighbors as inputs, and returns the integrated flux approximation (up to
        second order derivatives)
        """
        
        # Approximate the first order flux multiplier
        a = self.func_nn(u.unsqueeze(-1))
        
        # Apply the ReLU function for upwind scheme to prevent numerical
        # instability
        a_plus = th.relu(a[...,0])
        a_min = -th.relu(-a[...,0])
        
        
        ## Calculate fluxes at the left boundary of control volumes i
        
        # Calculate the flux at the left domain boundary
        left_bound_flux = (self.D*(self.stencil[0]*u[0,:] +
                            self.stencil[1]*self.BC[0,0]) -\
                            a_plus[0,:]/self.dx*(-self.stencil[0]*u[0,:] -
                            self.stencil[1]*self.BC[0,0])).unsqueeze(0)
                            
        # Calculate the fluxes between control volumes i and their left neighbors
        left_neighbors = self.D*(self.stencil[0]*u[1:,:] +
                            self.stencil[1]*u[:-1,:]) -\
                            a_plus[1:,:]/self.dx*(-self.stencil[0]*u[1:,:] -
                            self.stencil[1]*u[:-1,:])
        
        # Concatenate the left fluxes
        left_flux = th.cat((left_bound_flux, left_neighbors))
        
        ## Calculate fluxes at the right boundary of control volumes i
        
        # Calculate the flux at the right domain boundary
        right_bound_flux = (self.D*(self.stencil[0]*u[-1,:] +
                            self.stencil[1]*self.BC[1,0]) -\
                            a_min[-1,:]/self.dx*(self.stencil[0]*u[-1,:] +
                            self.stencil[1]*self.BC[1,0])).unsqueeze(0)
                 
        # Calculate the fluxes between control volumes i and their right neighbors
        right_neighbors = self.D*(self.stencil[0]*u[:-1,:] +
                            self.stencil[1]*u[1:,:]) -\
                            a_min[:-1,:]/self.dx*(self.stencil[0]*u[:-1,:] +
                            self.stencil[1]*u[1:,:])
        
        # Concatenate the right fluxes
        right_flux = th.cat((right_neighbors, right_bound_flux))
        
        # Calculate the flux at the bottom domain boundary
        bottom_bound_flux = (self.D*(self.stencil[0]*u[:,0] +
                            self.stencil[1]*self.BC[0,0]) -\
                            a_plus[:,0]/self.dy*(-self.stencil[0]*u[:,0] -
                            self.stencil[1]*self.BC[0,0])).unsqueeze(-1)
                            
        # Calculate the fluxes between control volumes i and their bottom neighbors
        bottom_neighbors = self.D*(self.stencil[0]*u[:,1:] +
                            self.stencil[1]*u[:,:-1]) -\
                            a_plus[:,1:]/self.dy*(-self.stencil[0]*u[:,1:] -
                            self.stencil[1]*u[:,:-1])
        
        # Concatenate the bottom fluxes
        bottom_flux = th.cat((bottom_bound_flux, bottom_neighbors), dim=1)
        
        ## Calculate fluxes at the top boundary of control volumes i
        
        # Calculate the flux at the top domain boundary
        top_bound_flux = (self.D*(self.stencil[0]*u[:,-1] +
                            self.stencil[1]*self.BC[1,0]) -\
                            a_min[:,-1]/self.dy*(self.stencil[0]*u[:,-1] +
                            self.stencil[1]*self.BC[1,0])).unsqueeze(-1)
                 
        # Calculate the fluxes between control volumes i and their top neighbors
        top_neighbors = self.D*(self.stencil[0]*u[:,:-1] +
                            self.stencil[1]*u[:,1:]) -\
                            a_min[:,:-1]/self.dy*(self.stencil[0]*u[:,:-1] +
                            self.stencil[1]*u[:,1:])
        
        # Concatenate the top fluxes
        top_flux = th.cat((top_neighbors, top_bound_flux), dim=1)
        
        
        # Integrate the fluxes at all boundaries of control volumes i
        flux = left_flux + right_flux + bottom_flux + top_flux
        
        return flux
    
    def state_kernel(self, t, u):
        """
        This function defines the state kernel for training, which takes the
        fluxes as inputs, and returns du/dt)
        """
        
        flux = self.flux_kernel(t, u)
        
        # Since there is no reaction term to be learned, du/dt = fluxes
        state = flux
        
        return state
    
    def forward(self, t, u):
        """
        This function integrates du/dt through time using the Neural ODE method
        """
        
        # The odeint function receives the function state_kernel that calculates
        # du/dt, the initial condition u[0], and the time at which the values of
        # u will be saved t
        pred = odeint(self.state_kernel, u[0], t)
        
        return pred
