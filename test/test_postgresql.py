import unittest
import math
from heat_battery.config import setup_local_config_path
from heat_battery.simulations.postgresql_project import Project
from heat_battery.simulations.jobs import generate_jobs
from heat_battery.simulations import sweep
from examples.Example_01.geometry import build_geometry
from examples.Example_01.model import Experiment_v1

class TestPostgresqlProject(unittest.TestCase):
    def setUp(self) -> None:
        
        try:
            setup_local_config_path('config.yaml')
            print(f"Running config database tests with config.yaml in cwd")
        except:
            setup_local_config_path('.github/github_test_config.yaml')
            print(f"Running config database tests in Github Actions with .github/github_test_config.yaml")
    
        self.project = Project('test_project_X', if_exists='override')
    def test_jobs_equivalence(self):
        p_inputs = sweep.ParameterGrid(dict(
            mesh_p = sweep.ParameterGrid(dict(
                dim=2,
                dir='examples/Example_01/meshes/',
            )),

            sim_p = sweep.ParameterGrid(dict(
                verbose=True,
                t_max = 100.0,
                result_dir='examples/Example_01/results/',
                probes_file='unsteady_postgress_test.csv',
            )),
        ))

        jobs = generate_jobs(
            sim_class=Experiment_v1, 
            mesh_builder=build_geometry,
            runner='solve_unsteady',
            group_name='test_group',
            p_grid=p_inputs,
        )

        self.project.add_jobs(jobs)
        remote_jobs = self.project.get_jobs()

        self.assertEqual(len(remote_jobs), len(jobs))
        for job_remote, job_local in zip(remote_jobs, jobs):
            print(job_remote)
            self.assertEqual(
                job_remote['signature'], job_local['signature'],
                f"Job signatures do not match: {job_remote['signature']} vs {job_local['signature']}"
            )
            self.assertEqual(
                job_remote['p_inputs']['mesh_p'], job_local['p_inputs']['mesh_p'],
                f"Mesh parameters do not match: {job_remote['p_inputs']['mesh_p']} vs {job_local['p_inputs']['mesh_p']}"
            )
            self.assertEqual(
                job_remote['p_inputs']['sim_p'], job_local['p_inputs']['sim_p'],
                f"Simulation parameters do not match: {job_remote['p_inputs']['sim_p']} vs {job_local['p_inputs']['sim_p']}"
            )
        job = jobs[0]
        job.generate_local_run_script()
        job.run()


    def tearDown(self) -> None:
        self.project.drop(fail_if_not_exists=True, cascade=True)

if __name__ == '__main__':
    unittest.main()