# Define torchlightning model
import lightning as L
from torch.utils.data import DataLoader
import torch
from .dataset import train_dataset, val_dataset, test_collate_fn, train_collate_fn
import numpy as np
import wandb
import evaluate

class ImageCaptioningModel(L.LightningModule):

    def __init__(self, model, processor, config):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model
        self.lr = self.config.get("lr", 1e-3)
        self.batch_size = self.config.get("batch_size", 32)
        self.processor = processor
        self.bleu_metric = evaluate.load("bleu", keep_in_memory=True)
        self.rouge_metric = evaluate.load("rouge", keep_in_memory=True)

        self.train_losses = []
        self.val_bleu = []
        self.val_rouge1 = []

    def on_train_epoch_start(self):
        self.train_losses = []

    def on_validation_epoch_start(self):
        self.val_bleu = []

    def training_step(self, batch_, batch_idx):

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

        self.log("train/loss", train_loss)
        self.log("train/loss_epoch", train_loss, on_epoch=True)
        self.log("train/avg_loss", np.mean(self.train_losses))
        self.log("train/avg_loss_epoch", np.mean(self.train_losses), on_epoch=True)

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

        bleu_score: float = self.bleu_metric.compute(references=labels, predictions=predictions)['bleu']
        self.val_bleu.append(bleu_score)
        rouge1_score: float = self.rouge_metric.compute(references=labels, predictions=predictions)['rouge1']
        self.val_rouge1.append(rouge1_score)

        self.log("val/bleu", bleu_score)
        self.log("val/bleu_epoch", bleu_score, on_epoch=True)
        self.log("val/avg_bleu", np.mean(self.val_bleu))


        self.log("val/rouge1", rouge1_score)
        self.log("val/epoch_rouge1", rouge1_score, on_epoch=True)
        self.log("val/avg_rouge1", np.mean(self.val_rouge1))


        # if the verbose flag is set, log the first 5 examples
        if self.config.get("verbose", False) and batch_idx <= 3:
            columns = ["global_step", "image", "ground_truth", "prediction"]
            datas = [
                    [self.global_step, wandb.Image(pixel_values[i]), labels[i], predictions[i]] for i in range(1)
                    ]

            self.logger.log_table(key="val/examples", columns=columns, data=datas)

        return predictions

    def on_train_epoch_end(self):
        print(f'Average Training Loss in EPOCH #{self.current_epoch}: {np.mean(self.train_losses)}')

        self.train_losses = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def train_dataloader(self):
        return DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=train_collate_fn,
                pin_memory=True,
                num_workers=self.config.get("num_workers", 2),
                )   

    def val_dataloader(self):
        return DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=test_collate_fn,
                pin_memory=True,
                num_workers=self.config.get("num_workers", 2),
                )
