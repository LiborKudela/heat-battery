import numpy as np
from mpi4py import MPI

class Optimizer:
    def __init__(self) -> None:

        # optimizer state
        self.g_value = None
        self.g_norm = None
        self.loss_value = None

        #TODO: add state tracker for visualization

    def get_k(self):
        return self.k.copy()

    def set_k(self, k):
        self.k = np.array(k)

class ADAM(Optimizer):
    def __init__(self, alpha=1e-3, beta_1=0.8, beta_2=0.8, eps=1e-1):
        
        # hyper parameters
        self.alpha = alpha
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps 

        # optimizer state
        self.g_w = 0.0
        self.g_s = 0.0
        self.j = 0

        super().__init__()

    def step(self, g):

        # update optimizer state
        self.j += 1
        self.g_norm = np.linalg.norm(g)
        self.g_w = self.beta_1*self.g_w + (1-self.beta_1)*g
        self.g_s = self.beta_2*self.g_s + (1-self.beta_2)*g**2
        self.g_w_hat = self.g_w/(1-self.beta_1**(self.j))
        self.g_s_hat = self.g_s/(1-self.beta_2**(self.j))
        update = self.alpha * self.g_w_hat/(np.sqrt(self.g_s_hat)+self.eps)
        return -update

# class Newton(Optimizer):
#     def __init__(self, alpha=None):
 
#         # hyper parameters
#         self.alpha = alpha
#         self.j = 0

#         super().__init__()

#     def step(self, g, l):
#         self.j += 1
#         self.g_norm = np.linalg.norm(g)
#         print(self.g_norm)
#         update = -self.alpha*(l/g)
#         return update
            
# TODO: Provide Newton solver
# TODO: Provide Broyden–Fletcher–Goldfarb–Shanno algorithm
# TODO: Provide L-BFGS

