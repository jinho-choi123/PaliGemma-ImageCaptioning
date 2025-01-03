import lightning as L
from lightning.pytorch.callbacks import Callback
from huggingface_hub import HfApi
from .config import config

api = HfApi()

class PushToHubCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        print(f"Train Epoch end: pushing model to hub, epoch {trainer.current_epoch}")
        pl_module.model.push_to_hub(config.get("hf_checkpoint_repo_id", "ball1433/Paligemma-ImageCaptioning"), commit_message=f"Training in progress: epoch {trainer.current_epoch}")


