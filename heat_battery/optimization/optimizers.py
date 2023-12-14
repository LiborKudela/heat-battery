import numpy as np
from mpi4py import MPI

class Optimizer:
    def __init__(self, loss=None, grad=None, grad_returns_loss=False, k0=None) -> None:
        self.loss = loss
        self.grad = grad
        self.grad_returns_loss = grad_returns_loss

        # optimizer state
        self.future_update = None
        self.g_value = None
        self.g_norm = None
        self.loss_value = None
        if k0 is not None:
            self.k = k0.copy()

        #TODO: add state tracker for visualization

    def get_k(self):
        return self.k.copy()

    def set_k(self, k):
        self.k = np.array(k)

    def gradient_finite_differences(self, k, perturbation=1e-8, return_loss=True):
        org_loss_value = self.loss(k)
        k_pert = k.copy()
        g = []
        for i in range(len(k)):
            k_pert[i] += perturbation
            pert_loss_value = self.loss(k_pert)
            k_pert[i] -= perturbation
            g_err = (pert_loss_value-org_loss_value)/perturbation
            g.append(g_err)
        if return_loss:
            return np.array(g), org_loss_value
        else:
            return np.array(g)
        
    def objective(self, k):
        if self.grad is None:
            return self.gradient_finite_differences(k, return_loss=True)
        else:
            if self.grad_returns_loss:
                return self.grad(k)
            else:
                return self.grad(k), self.loss(k)

    def print_state(self):
        if MPI.COMM_WORLD.rank == 0:
            print(f"step: {self.j}" , 
                  f"loss: {self.loss_value}", 
                  f"g_norm: {self.g_norm}",
                  )

class ADAM(Optimizer):
    def __init__(self, loss=None, grad=None, grad_returns_loss=False, k0=None, k_min=None, k_max=None, alpha=1e-3, beta_1=0.8, beta_2=0.8, eps=1e-1):
        assert loss is not None or grad_returns_loss, "loss function must be given unless gradient also returns loss"
        
        # hyper parameters
        self.alpha = alpha
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps 


        self.a_min = k_min or -np.inf
        self.a_max = k_max or np.inf

        # optimizer state
        self.g_w = 0.0
        self.g_s = 0.0
        self.j = 0

        super().__init__(loss=loss, grad=grad, grad_returns_loss=grad_returns_loss, k0=k0)

    def step(self):
        self.j += 1
        if self.j > 1:
            self.k -= self.future_update
            np.clip(self.k, self.a_min, self.a_max, self.k)
        g, l = self.objective(self.k)

        # update optimizer state
        self.g_norm = np.linalg.norm(g)
        self.g_value = g
        self.loss_value = l
        self.g_w = self.beta_1*self.g_w + (1-self.beta_1)*g
        self.g_s = self.beta_2*self.g_s + (1-self.beta_2)*g**2
        self.g_w_hat = self.g_w/(1-self.beta_1**(self.j))
        self.g_s_hat = self.g_s/(1-self.beta_2**(self.j))
        self.future_update = self.alpha * self.g_w_hat/(np.sqrt(self.g_s_hat)+self.eps)

    def print_state(self):
        if MPI.COMM_WORLD.rank == 0:
            print(f"step: {self.j}" , 
                  f"loss: {self.loss_value}", 
                  f"g_norm: {self.g_norm}",
                  f"alpha: {self.alpha}",
                  )
            
    def print_k(self):
        if MPI.COMM_WORLD.rank == 0:
            print(self.k)
            
# TODO: Provide Newton solver
# TODO: Provide Broyden–Fletcher–Goldfarb–Shanno algorithm
# TODO: Provide L-BFGS

