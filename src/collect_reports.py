import os
import pandas as pd

from dotenv import load_dotenv
from preprocessing import utils

load_dotenv()
project_dir = os.getenv('project_dir')
subjects = utils.get_subject_id('all')

df = pd.DataFrame(
    columns=[
    'subject',
    'session',
    'N_trials_remembered_raw',
    'N_trials_forgotten_raw',
    'N_trials_remembered_clean',
    'N_trials_forgotten_clean'
    ]
)

for subject in subjects:
    for session in utils.get_subject_sessions(subject):
        
        try:
            report = utils.get_report(subject, session)
            df_session = pd.DataFrame.from_dict(report, orient='index').T
            df_session['subject'] = subject
            df_session['session'] = session
            df = pd.concat([df, df_session])
        except:
            continue

df.to_csv(f'{project_dir}/results/report.csv', index=False)