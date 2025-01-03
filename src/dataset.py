# Preprocess the data
# and create pytorch dataset
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
import json
from transformers import AutoProcessor
from .config import config

# define a processor for paligemma
processor = AutoProcessor.from_pretrained(config.get("pretrained_repo_id"))


def train_collate_fn(batch):
    # batch is a list of tuples
    # each tuple is (image, label)
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    prompts = ["generated caption:" for item in batch]

    # current labels are json format 
    # {"prompt": "<image caption>"}
    # extract the caption part fron json
    labels = [label["prompt"] for label in labels]

    inputs = processor(text=prompts, images=images, suffix=labels, return_tensors="pt", padding=True, truncation=True, max_length=config.get("max_length"))

    input_ids = inputs["input_ids"]
    token_type_ids = inputs["token_type_ids"]
    attention_mask = inputs["attention_mask"]
    pixel_values = inputs["pixel_values"]
    labels = inputs["labels"]

    return input_ids, token_type_ids, attention_mask, pixel_values, labels

def test_collate_fn(batch):
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    prompts = ["generated caption:" for item in batch]

    labels = [label["prompt"] for label in labels]
    
    inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True, truncation=True, max_length=config.get("max_length"))

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    pixel_values = inputs["pixel_values"]

    return input_ids, attention_mask, pixel_values, labels

class ImageCaptionDataset(Dataset):
    def __init__(self, data_dir: Path, transforms=None):
        """
        For initializing dataset, pass the data directory(abs path) and transforms
        """
        self.transforms = transforms
        self.data_dir = data_dir

        # load the data_names
        self.data = []
        for item in data_dir.glob("*.json"):
            self.data.append(item.stem)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # load image and caption
        image_path = self.data_dir / f"{self.data[idx]}.jpg"
        caption_path = self.data_dir / f"{self.data[idx]}.json"

        image = Image.open(image_path).convert("RGB")
        with open(caption_path, "r") as f:
            caption = json.load(f)
            caption = caption["prompt"]

        if self.transforms:
            image = self.transforms(image)

        return image, caption

# Define train_dataset and test_dataset

dataset = ImageCaptionDataset(data_dir=Path("data/"))
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(0.8*len(dataset)), len(dataset)-int(0.8*len(dataset))])

# Test the dataset
# print(len(train_dataset))
# print(len(val_dataset))
# print(f"Image: {train_dataset[0][0].show()}")
# print(f"Caption: {train_dataset[0][1]}")
