

class Optimizer:
    def __init__(self, g) -> None:
        pass

    def optimize(self, loss, k0=None, max_iter=1000, alpha=0.01, k0=None):


        for j in range(max_iter): 
            if MPI.COMM_WORLD.rank == 0:
                print(f"iter {j}")
            
            for sub_iter, ic in enumerate(idx_couples):
                m = ic[0]
                k = ic[1]
                nominal = self.sim.mats[m].k.get_value(k)
                self.steady_state.update()
                switched_sign = False
                if MPI.COMM_WORLD.rank == 0:
                    abs_e = self.steady_state.total_abs_error.copy()
                    sqr_e = self.steady_state.total_square_error.copy()
                    print(f"  sub iter: {sub_iter}", f"abs_err: {abs_e}" , f"sqr_err: {sqr_e}")
                local_d = np.abs(nominal)*alpha
                for i in range(2):
                    prev_e = np.sqrt(self.steady_state.total_max_error.copy())
                    v = self.sim.mats[m].k.get_value(k)
                    self.sim.mats[m].k.set_value(k, v+local_d)

                    try:
                        self.steady_state.update()
                        new_e = np.sqrt(self.steady_state.total_max_error.copy())
                    except:
                        new_e = np.inf
                        
                    if new_e > prev_e:
                        self.sim.mats[m].k.set_value(k, v)
                        if switched_sign:
                            break
                        else:
                            local_d = -local_d
                            switched_sign = True
                            continue
            res = [mat.k.get_values() for mat in self.sim.mats]
            if MPI.COMM_WORLD.rank == 0:
                print(res)
