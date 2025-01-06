
config = {
        "pretrained_repo_id": "google/paligemma-3b-pt-224",
        "max_length": 256,
        "max_new_tokens": 128,
        "lora_r": 8,
        "wandb_project": "paligemma-image-captioning",
        "hf_checkpoint_repo_id": "ball1433/Paligemma-ImageCaptioning",
        "batch_size": 8, # for L4 GPU: 8, T4: 2
        "verbose": True,
        "lr": 1e-4, 
        }
