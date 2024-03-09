import argparse
import preprocess

parser = argparse.ArgumentParser()
for arg in ['--subject', '--session', '--stage']:
    parser.add_argument(arg)
args = parser.parse_args()


if args.stage == 'raw':
    preprocess.preprocess_raw(args.subject, args.session)

if args.stage == 'epochs':
    preprocess.preprocess_epochs(args.subject, args.session)

if args.stage == 'dataframe':
    preprocess.preprocess_dataframe(args.subject, args.session)
