# Actual training happens here

from peft.peft_model import PeftModel, PeftModelForCausalLM
from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict
import torch
torch.manual_seed(42)
from transformers import BitsAndBytesConfig, PaliGemmaForConditionalGeneration
from peft import get_peft_model, LoraConfig, AutoPeftModelForCausalLM
from .config import config
from lightning.pytorch.loggers import WandbLogger
from .model import ImageCaptioningModel
from .dataset import processor
from .train_callbacks import PushToHubCallback
import lightning as L
from lightning.pytorch.tuner import Tuner


# define wandb logger
wandb_logger = WandbLogger(project=config.get("wandb_project"))

bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        )
lora_config = LoraConfig(
        r=config.get("lora_r", 8),
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
        )

model = PaliGemmaForConditionalGeneration.from_pretrained(config.get("pretrained_repo_id"), quantization_config=bnb_config)

if config.get("load_lora", False):
    # model = PaliGemmaForConditionalGeneration.from_pretrained(config.get("hf_checkpoint_repo_id"), quantization_config=bnb_config)
    model = PeftModel.from_pretrained(model, config.get("hf_checkpoint_repo_id"), is_trainable=True)

else:
    model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

model_module = ImageCaptioningModel(model, processor, config)


trainer = L.Trainer(
        accumulate_grad_batches=4,
        max_epochs=50,
        gradient_clip_val=1.0,
        num_sanity_val_steps=5,
        logger=wandb_logger,
        callbacks=[PushToHubCallback()],
        # val_check_interval=0.1,
        )

# tuner = Tuner(trainer)

# tuner.scale_batch_size(model_module)
# tuner.lr_find(model_module)

trainer.fit(model_module)
