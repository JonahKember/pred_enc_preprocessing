import os
import pandas as pd

from dotenv import load_dotenv

load_dotenv()
project_dir = os.getenv('project_dir')


def analyze_report():
    # Analyze preprocessing report.
    report = pd.read_csv(f'{project_dir}/results/report.csv')

    total = report.N_trials_remembered_raw.sum() + report.N_trials_forgotten_raw.sum()
    total_clean = report.N_trials_remembered_clean.sum() + report.N_trials_forgotten_clean.sum()

    trials_R = report.N_trials_remembered_raw.sum()
    trials_F = report.N_trials_forgotten_raw.sum()

    trials_R_clean = report.N_trials_remembered_clean.sum()
    trials_F_clean = report.N_trials_forgotten_clean.sum()

    print(f'\nPreprocessing report:\n\nTotal trials:\nRaw = {total:,}\nClean = {total_clean:,}')
    print(f'    WORD_R: {trials_R_clean:,} ({100*(trials_R_clean/total_clean):.2f}% of all trials)')
    print(f'    WORD_F: {trials_F_clean:,} ({100*(trials_F_clean/total_clean):.2f}% of all trials)\n')

    print('Artifact rejection by condition:')
    print(f'WORD_R: {100*(trials_R_clean/trials_R):.2f}% kept after cleaning')
    print(f'WORD_F: {100*(trials_F_clean/trials_F):.2f}% kept after cleaning\n')


if __name__ == '__main__':
    analyze_report()