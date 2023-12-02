from typing import Any

import lightning.pytorch as pl
import torch
import torchmetrics
from omegaconf import DictConfig
from torch import softmax
from torch.optim import Adam


class MyModel(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        self.linear_1 = torch.nn.Linear(cfg.model.input_dim, cfg.model.output_dim)
        # self.linear_2 = torch.nn.Linear(cfg.model.hidden_dim1, cfg.model.hidden_dim2)
        # self.linear_3 = torch.nn.Linear(cfg.model.hidden_dim2, cfg.model.hidden_dim3)
        # self.linear_4 = torch.nn.Linear(cfg.model.hidden_dim3, cfg.model.output_dim)

        self.act = torch.nn.ReLU()

        self.batch_norm_1 = torch.nn.BatchNorm1d(cfg.model.input_dim)
        # self.batch_norm_2 = torch.nn.BatchNorm1d(cfg.model.hidden_dim2)
        # self.batch_norm_3 = torch.nn.BatchNorm1d(cfg.model.hidden_dim3)

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.f1_fn = torchmetrics.classification.F1Score(task=cfg.model.f1_task, num_classes=cfg.model.output_dim)

    def forward(self, x):
        x = self.batch_norm_1(x)
        x = self.act(x)
        x = self.linear_1(x)

        # x = self.linear_2(x)
        # x = self.batch_norm_2(x)
        # x = self.act(x)
        #
        # x = self.linear_3(x)
        # x = self.batch_norm_3(x)
        # x = self.act(x)
        #
        # x = self.linear_4(x)
        # x = self.act(x)

        return softmax(x, dim=1)

    def configure_optimizers(self) -> Any:
        return Adam(self.parameters(), lr=self.cfg.model.lr)

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
        f1 = self.f1_fn(predicted, labels).item()
        self.log_dict({"val_loss": loss, "val_acc": val_acc, "val_f1": f1},
                      on_step=False, on_epoch=True, prog_bar=True)
        return loss
