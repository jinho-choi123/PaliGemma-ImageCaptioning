from .config import config
from transformers import BitsAndBytesConfig, PaliGemmaForConditionalGeneration
from peft import PeftModel
import torch
from PIL import Image
from transformers import AutoProcessor

# define device
device = torch.device("cuda")

# define a processor for paligemma
processor = AutoProcessor.from_pretrained(config.get("pretrained_repo_id"))

# IMG_PATH = "path/to/image.jpg"
IMG_PATH = "examples/example.jpg"
images = [Image.open(IMG_PATH)]
prompts = ["<image>Caption for the image is "]



model = PaliGemmaForConditionalGeneration.from_pretrained(config.get("pretrained_repo_id"))
print(f"Loaded PaliGemma model...")

# set the is_trainable flag as False because we are doing inference-only
model = PeftModel.from_pretrained(model, config.get("hf_checkpoint_repo_id"), is_trainable=False)
print(f"Loaded peft adapter for PaliGemma model...")

model.to(device)

with torch.no_grad():
    inference_inputs = processor(text=prompts, images=images, return_tensors="pt")
    print(f"processing the input image completed...")
    input_ids = inference_inputs["input_ids"]
    attention_mask = inference_inputs["attention_mask"]
    pixel_values = inference_inputs["pixel_values"]

    generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            max_new_tokens=config.get("max_new_tokens")
            )

    print(f"Finished generating the predictions...")

    predictions = processor.batch_decode(generated_ids[:, input_ids.size(1)+1:], skip_special_tokens=True)
    
    images[0].show()
    print(f"{prompts[0]}{predictions[0]}")


