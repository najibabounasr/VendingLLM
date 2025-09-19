import subprocess
import os
from dvc.repo import Repo
import configparser
from local_settings import settings
from dagshub import get_repo_bucket_client
def run_dvc_command(command):
    try:
        result = subprocess.run(command, shell=True, check=True, text=True)
        if result.returncode != 0:
            raise Exception(f"DVC command failed: {command}")
    except subprocess.CalledProcessError as e:
        print(f"Command '{command}' failed with error: {e}")
        raise
def load_dvc_config():
    config_path = os.path.join('.dvc', 'config')
    config = configparser.ConfigParser()
    config.read(config_path)
    return config

def check_remote_config(config):
    if config is None or not isinstance(config, configparser.ConfigParser):
        return False
    if 'remote "origin"' in config.sections():
        return True
    return False

def verify_dvc_remote():
    try:
        repo = Repo()
        config = load_dvc_config()

        if not check_remote_config(config):
            raise RuntimeError("DVC remote 'origin' is not configured. Please check your DVC remote settings.")
        
        print("DVC remote 'origin' is correctly configured.")
    except Exception as e:
        raise RuntimeError(f"Failed to verify DVC remote configuration: {e}")

def dagshub_initialization():
    # Set up DVC remote storage (DAGsHub)
    os.environ['DVC_REMOTE_URL'] = 'https://dagshub.com/najibabounasr/MacroEconomicAPI.dvc'
    os.environ['DVC_REMOTE_USER'] = 'najibabounasr'
    os.environ['DVC_REMOTE_PASSWORD'] = 'fbaccfb8cf4e8d2d195cd05e9a53dbfe32323695'

    # Initialize DVC if not already initialized
    if not os.path.isdir('.dvc'):
        run_dvc_command("dvc init")

    repo = Repo()
    config = repo.config.read()
    remotes = config.get('remote', {})

    # Check if the remote 'origin' already exists
    if 'origin' not in remotes:
        run_dvc_command(f"dvc remote add -d origin {os.environ['DVC_REMOTE_URL']}")
    else:
        existing_url = remotes['origin']['url']
        if existing_url != os.environ['DVC_REMOTE_URL']:
            print(f"Updating 'origin' URL from {existing_url} to {os.environ['DVC_REMOTE_URL']}")
            run_dvc_command(f"dvc remote modify origin url {os.environ['DVC_REMOTE_URL']}")

    # Set 'core' remote to 'origin'
    run_dvc_command("dvc config core.remote origin")

    # Set or update authentication for the remote
    run_dvc_command(f"dvc remote modify origin auth basic")
    run_dvc_command(f"dvc remote modify origin user {os.environ['DVC_REMOTE_USER']}")
    run_dvc_command(f"dvc remote modify origin password {os.environ['DVC_REMOTE_PASSWORD']}")

    # Confirm remote setup
    print(f"DVC remote 'origin' set to {os.environ['DVC_REMOTE_URL']}")

def upload_to_dagshub(filename, key):
    # Extract repo name from settings if necessary
    repo_name = settings['dagshub_repo']  # Assuming settings['dagshub_repo'] is in format "username/repo_name"
    
    # Initialize Dagshub client
    s3 = get_repo_bucket_client(repo_name)
    
    # Upload file to Dagshub storage
    s3.upload_file(
        Bucket=repo_name.split('/')[1],
        Filename=filename,
        Key=key
    )

# Initialize DVC for DAGsHub repository
dagshub_initialization()