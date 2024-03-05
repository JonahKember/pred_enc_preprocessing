# Download the necessary external data.

# Activate the environment.
source .env
source ENV/bin/activate
aws s3 sync --no-sign-request s3://openneuro.org/ds004395 $project_dir/data/raw/ds004395

# Donwload MNE data and HCPEx data???????
