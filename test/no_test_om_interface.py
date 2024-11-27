import unittest

class TestOMInterface(unittest.TestCase):

    def setUp(self) -> None:
        
        from heat_battery.simulations.om_interface import OMFMU2model
        import numpy as np

        self.test_fmu = OMFMU2model(
            'Single_pipe_array.fmu', 
            cvodemaxNumSteps=10000,
            maxStep=0.1,
            inputs=dict(
                T=lambda t: 293.15 if t <= 1.0 else 323.15,
                m_flow=lambda t: 0.0+0.001*t,
                T_pipe_surface=lambda t: np.full(10, 293.15) if t <= 5.0 else np.full(10, 323.15),
                ),
            events=[5.0],
            outputs_matches={
                'pipe': 'pipe.heatTransfer.Ts\[[0-9]*\]',
                'pipe1': 'pipe1.heatTransfer.Ts\[[0-9]*\]',
                'pipe2': 'pipe2.heatTransfer.Ts\[[0-9]*\]',
                }
            )
        
    def test_simulation(self):
        self.test_fmu.instantiate(debug_logging=False)
        self.test_fmu.simulate(stop_time=100.0, interval=5.0, verbose=True)
        self.test_fmu.output.get_outputs('pipe')

        self.assertTrue(True)
    
    def tearDown(self):
        self.test_fmu.destroy()

if __name__ == '__main__':
    unittest.main()
