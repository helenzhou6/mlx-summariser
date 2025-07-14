import torch
import wandb
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv(override=True)

def get_device():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")
    return device

def init_wandb(config={}):
    default_config = {
        "learning_rate": 0.001,
        "architecture": "RLHF",
        "dataset": "huggingface-reddit",
        "epochs": 5,
    }
    # Start a new wandb run to track this script.
    return wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity=os.environ.get("WANDB_ENTITY", "mlx-summariser"),
        # Set the wandb project where this run will be logged.
        project=os.environ.get("WANDB_PROJECT", "default"),
        # Track hyperparameters and run metadata.
        config={**default_config, **config},
    )

def save_artifact(artifact_name, artifact_description, file_extension='pt', type="model"):
    artifact = wandb.Artifact(
        name=artifact_name,
        type=type,
        description=artifact_description
    )
    artifact.add_file(f"./data/{artifact_name}.{file_extension}")
    wandb.log_artifact(artifact)

def load_artifact_path(artifact_name, version="latest", file_extension='pt'):
    artifact = wandb.use_artifact(f"{artifact_name}:{version}")
    directory = artifact.download()
    return f"{directory}/{artifact_name}.{file_extension}"
