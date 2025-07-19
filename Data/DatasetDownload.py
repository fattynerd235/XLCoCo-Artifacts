import os
from kaggle.api.kaggle_api_extended import KaggleApi

# ----------- Configuration -----------
kaggle_json_path = './kaggle.json'  # Adjust if you're storing it in a custom location
dataset_slug = 'fattynerd/xlcoco-artifacts'
download_dir = './CSVFiles'
unzip_after_download = True
# --------------------------------------

# Setup environment
if not os.path.exists(download_dir):
    os.makedirs(download_dir)

# Set Kaggle API key environment variable
os.environ['KAGGLE_CONFIG_DIR'] = os.path.dirname(kaggle_json_path)

# Initialize Kaggle API
api = KaggleApi()
api.authenticate()

# Download dataset (private allowed if you own/have access)
print(f"Downloading: {dataset_slug} ...")
api.dataset_download_files(dataset_slug, path=download_dir, unzip=unzip_after_download)
print(f"Downloaded to: {download_dir}")
