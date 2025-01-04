# Define torchlightning model
import lightning as L
from torch.utils.data import DataLoader
import torch
from .dataset import train_dataset, val_dataset, test_collate_fn, train_collate_fn
import numpy as np

class ImageCaptioningModel(L.LightningModule):

    def __init__(self, model, processor, config):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model
        self.lr = self.config.get("lr", 1e-3)
        self.batch_size = self.config.get("batch_size", 32)
        self.processor = processor

        self.train_losses = []

    def training_step(self, batch_,  batch_idx):

        input_ids, token_type_ids, attention_mask, pixel_values, labels = batch_

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
            labels=labels
            )

        train_loss = outputs.loss
        self.train_losses.append(train_loss.item())

        self.log("train/step_loss", train_loss)
        self.log("train/epoch_loss", train_loss, on_epoch=True)

        return train_loss
    
    def validation_step(self, batch, batch_idx):

        input_ids, attention_mask, pixel_values, labels = batch

        generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                max_new_tokens=self.config.get("max_new_tokens")
                )
        predictions = self.processor.batch_decode(generated_ids[:, input_ids.size(1)+1:], skip_special_tokens=True)

        if self.config.get("verbose", False) and batch_idx == 1:
            self.log("val/predictions", predictions[0])
            self.log("val/labels", labels[0])

        return predictions

    def on_train_epoch_end(self):
        print(f'Average Training Loss in EPOCH #{self.current_epoch}: {np.mean(self.train_losses)}')

        self.train_losses = []

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def train_dataloader(self):
        return DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=train_collate_fn
                )   

    def val_dataloader(self):
        return DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=test_collate_fn
                )
