#######################################
# Download the necessary external data.
#######################################

# Activate the environment.
source .env
source ENV/bin/activate
aws s3 sync --no-sign-request s3://openneuro.org/ds004395 $project_dir/data/raw/ds004395

# Download fsaverage data, if non-existent.
python - <<END
import mne
mne.datasets.fetch_fsaverage('$project_dir/data/external')
END

# Download extended HCP-MMPv1 atlas.
git clone https://github.com/wayalan/HCPex $project_dir/data/external/HCPex