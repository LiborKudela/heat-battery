"""MPI worker: pull scheduled jobs from PostgreSQL and run them sequentially."""

import argparse

from heat_battery.simulations.postgresql_project import Project, print_rank_0


def main():
    parser = argparse.ArgumentParser(
        description="Run up to N jobs from the simulation queue (use with mpirun)."
    )
    parser.add_argument(
        "-n",
        "--num-jobs",
        type=int,
        default=1,
        help="Maximum jobs to dequeue and run one after another (default: 1).",
    )
    parser.add_argument(
        "-p",
        "--project",
        default="project_example_05",
        help="Database project name.",
    )
    parser.add_argument(
        "-fp",
        "--force-priority",
        type=int,
        default=None,
        help="Force priority of the job to run instead of picking next one, runs single job (default: None).",
    )
    args = parser.parse_args()

    project = Project(args.project)

    if args.force_priority is not None:
        job = project.get_next_scheduled_job(priority=args.force_priority)
        print_rank_0(f"[worker] force-priority={args.force_priority}: {job}")
        if job is None:
            print_rank_0("[worker] no SCHEDULED/FAILED*/INTERRUPTED job with that priority; exiting.")
            return
        job.run()
        return

    for i in range(args.num_jobs):
        job = project.get_next_scheduled_job()
        print_rank_0(f"[worker] slot {i + 1}/{args.num_jobs}: {job}")
        if job is None:
            print_rank_0("[worker] no SCHEDULED/FAILED*/INTERRUPTED job left; exiting loop.")
            break
        job.run()


if __name__ == "__main__":
    main()
