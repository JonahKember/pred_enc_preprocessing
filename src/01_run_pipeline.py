import os
import argparse
import subprocess

from glob import glob
from dotenv import load_dotenv

from preprocessing import utils
from preprocessing.config import job_params

load_dotenv()
project_dir = os.getenv('project_dir')

parser = argparse.ArgumentParser()
parser.add_argument('--create', action='store_true')
parser.add_argument('--run', action='store_true')
args = parser.parse_args()


def create_jobs():
    '''For each session, write an SBATCH script to '/jobs' with the specified preprocessing stage.'''

    for subject in utils.get_subject_id('all'):
        for session in utils.get_subject_sessions(subject):

            processed = utils.has_been_processed(subject, session)

            if not processed[job_params['stage']]:
                utils.create_sbatch(
                    subject,
                    session,
                    stage=job_params['stage'],
                    hours=job_params['hours'],
                    minutes=job_params['minutes'],
                    mem_per_cpu=job_params['mem_per_cpu']
                )


def run_jobs():
    '''Submit all the jobs currently in /jobs to the scheduler.'''

    jobs = glob(f'{project_dir}/jobs/*{job_params["stage"]}*')

    for job in jobs[:job_params['n_jobs']]:
        subprocess.Popen(['sbatch', job])


if __name__ == '__main__':
    if args.create: create_jobs()
    if args.run: run_jobs()