import numpy as np
from mpi4py import MPI
#from .derivatives import finite_diferences

class Optimizer:
    def __init__(self) -> None:

        # optimizer state
        self.g_value = None
        self.g_norm = None
        self.loss_value = None

        #TODO: add state tracker for visualization

    def optimise(self, k0, tol=1e-6, max_iter=100, verbose=True) -> np.ndarray:
        k = k0.copy()
        for i in range(max_iter):
            k = self.step(k)
            if MPI.COMM_WORLD.rank == 0 and verbose:
                print(f"{self.__class__.__name__} step {i+1}: k: {k}", f"loss: {self.l}")
            if self.l < tol:
                break
        return k

class ADAM(Optimizer):
    def __init__(self, grad, alpha=1e-3, beta_1=0.8, beta_2=0.8, eps=1e-1, 
                 alpha_decay=1.0):

        # callables
        self.grad = grad
        
        # hyper parameters
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps 

        # optimizer state
        self.g = None
        self.l = None
        self.g_w = 0.0
        self.g_s = 0.0
        self.j = 0

        super().__init__()

    def step(self, k):
        # update optimizer state
        self.g, self.l = self.grad(k)
        self.j += 1
        self.g_norm = np.linalg.norm(self.g)
        self.g_w = self.beta_1*self.g_w + (1-self.beta_1)*self.g
        self.g_s = self.beta_2*self.g_s + (1-self.beta_2)*self.g**2
        self.g_w_hat = self.g_w/(1-self.beta_1**(self.j))
        self.g_s_hat = self.g_s/(1-self.beta_2**(self.j))
        update = -self.alpha * self.g_w_hat/(np.sqrt(self.g_s_hat)+self.eps)
        k = k+update
        self.alpha *= self.alpha_decay
        return k
            
class GaussNewton(Optimizer):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def step(self, j, err):
        update = -self.alpha*np.linalg.pinv(j)@err
        return update
    
def line_search(loss, k0, direction, alpha0=1.0, tol=1e-6, max_it=np.inf, verbose=True):
    normed_dir = direction/np.linalg.norm(direction)
    prev_l = loss(k0)
    k = k0.copy()
    alpha = alpha0
    i = 0
    while True:

        # evaluate model
        k += alpha*normed_dir
        try:
            l = loss(k)
        except:
            l = np.inf

        # if not better revert, reverse dir, downsize alpha, 
        # if alpha too small stop 
        if l > prev_l:
            k -= alpha*normed_dir
            alpha *= -0.5
            if abs(alpha) < tol:
                if MPI.COMM_WORLD.rank == 0 and verbose:
                    print(f"Line search: k: {k}", f"alpha: {alpha}", f"loss: {l}")
                break
            continue
        else:
            # otherwise keep steping and show progress
            prev_l = l
            if MPI.COMM_WORLD.rank == 0 and verbose:
                print(f"Line search: k: {k}", f"alpha: {alpha}", f"loss: {l}")
            i += 1
            if i > max_it:
                break
    return k, l

class GaussNewtonDescent(Optimizer):
    def __init__(self, loss, grad, jac, alpha=2.0, max_it=np.inf, sigma_lim=1e-15):
        self.loss = loss
        self.grad = grad
        self.jac = jac
        self.alpha = alpha
        self.max_it = max_it
        self.sigma_lim = sigma_lim

    def step(self, k):

        j = self.jac(k)
        s = np.linalg.svd(j, compute_uv=False)
        j_rank = len(s)
        j_eff_rank = len(s >= self.sigma_lim)
        assert j_rank == len(k), (
            f"Jacobian has insufficient TRUE rank:\n" 
            f"The TRUE rank is {j_rank}, but the problem requires rank {len(k)}.\n"
            f"The Jacobian does not contain enough data.\n"
            f"Unique solution is not posible, descend direction is not guaranteed.\n"
        )

        assert j_eff_rank == len(k), (
            f"Jacobian has insufficient EFFECTIVE rank:\n" 
            f"The EFFECTIVE rank is {j_eff_rank}, but the problem requires {len(k)}.\n"
            f"There is not enough variance in the jacobian data.\n"
            f"In order to ignore this error, lower the sigma_lim cutoff limit.\n"
            f"The insufficient singular values (where s<{self.sigma_lim}) are:\n"
            f"{s[s< self.sigma_lim]}"
        )
        
        g, l = self.grad(k)
        update_dir = -np.linalg.inv(j@j.T)@g #TODO: use linsolve instead
        k, self.l = line_search(self.loss, k, update_dir, alpha0=self.alpha, 
                        max_it=self.max_it)
        return k

class RandomDirectionSearch(Optimizer):
    def __init__(self, loss, alpha=0.01, samples=10):
        self.loss = loss
        self.alpha = alpha
        self.samples = samples

    def step(self, k):
        if self.v is None:
            self.v = np.zeros_like(k)
        k0 = k.copy()
        prev_l = self.loss(k0)
        new_dir = None
        for i in range(self.samples):
            eps = np.random.normal(0.0, 0.0001, len(k))
            eps = MPI.COMM_WORLD.bcast(eps)
            new_k = k0+self.alpha*eps
            new_l = self.loss(new_k)
            if new_l < prev_l:
                new_dir = eps
        if new_dir is not None:
            k, l = line_search(self.loss, k, new_dir, alpha0=self.alpha)
        return k
    
class GradientSearch(Optimizer):
    def __init__(self, loss, grad, alpha=0.01):
        self.loss = loss
        self.grad = grad
        self.alpha = alpha

    def step(self, k):
        g = self.grad(k)[0]
        k, l = line_search(self.loss, k, -g, alpha0=self.alpha, tol=1e-12)
        return k

# TODO: Provide Newton solver
# TODO: Provide Broyden–Fletcher–Goldfarb–Shanno algorithm
# TODO: Provide L-BFGS

