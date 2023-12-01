from typing import Any

import lightning.pytorch as pl
import torch
from omegaconf import DictConfig
from torch import softmax
from torch.optim import Adam


class MyModel(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        self.linear_1 = torch.nn.Linear(cfg.model.input_dim, cfg.model.hidden_dim1)
        self.linear_2 = torch.nn.Linear(cfg.model.hidden_dim1, cfg.model.hidden_dim2)
        self.linear_3 = torch.nn.Linear(cfg.model.hidden_dim2, cfg.model.hidden_dim3)
        self.linear_4 = torch.nn.Linear(cfg.model.hidden_dim3, cfg.model.output_dim)

        self.act = torch.nn.ReLU()

        self.batch_norm_1 = torch.nn.BatchNorm1d(cfg.model.hidden_dim1)
        self.batch_norm_2 = torch.nn.BatchNorm1d(cfg.model.hidden_dim2)
        self.batch_norm_3 = torch.nn.BatchNorm1d(cfg.model.hidden_dim3)

        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.linear_1(x)
        x = self.batch_norm_1(x)
        x = self.act(x)

        x = self.linear_2(x)
        x = self.batch_norm_2(x)
        x = self.act(x)

        x = self.linear_3(x)
        x = self.batch_norm_3(x)
        x = self.act(x)

        x = self.linear_4(x)
        x = self.act(x)

        return softmax(x, dim=1)

    def configure_optimizers(self) -> Any:
        return Adam(self.parameters(), lr=self.cfg.model.lr)

    # def on_before_optimizer_step(self, optimizer):
    #     self.log_dict(pl.utilities.grad_norm(self, norm_type=2))
    #     super().on_before_optimizer_step(optimizer)

    def training_step(self, batch: Any, batch_idx: int, dataloader_idx=0):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, labels)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        inputs, labels = batch
        outputs: torch.Tensor = self(inputs)
        loss = self.loss_fn(outputs, labels)
        predicted = torch.argmax(outputs, dim=1)
        val_acc = torch.sum(labels == predicted).item() / (len(predicted) * 1.0)
        self.log_dict({"val_loss": loss, "val_acc": val_acc}, on_step=False, on_epoch=True, prog_bar=True)
        return loss
