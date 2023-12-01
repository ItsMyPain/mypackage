import hydra
import lightning.pytorch as pl
import lightning.pytorch.loggers as pl_loggers
import torch
from omegaconf import DictConfig

from my_package.data import MyDataModule
from my_package.model import MyModel


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def train(cfg: DictConfig):
    pl.seed_everything(cfg.data.seed)

    data_module = MyDataModule(
        csv_path=cfg.data.name,
        target=cfg.data.target,
        val_size=cfg.data.val_size,
        seed=cfg.data.seed,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers
    )

    model = MyModel(cfg)

    loggers = [
        pl_loggers.MLFlowLogger(
            experiment_name=cfg.mlflow.experiment_name,
            tracking_uri=cfg.mlflow.uri,
            log_model=True
        )
    ]

    callbacks = [
        pl.callbacks.LearningRateMonitor(logging_interval="epoch")
    ]

    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        precision=cfg.training.precision,
        accumulate_grad_batches=cfg.training.accum_grad_batches,
        val_check_interval=cfg.training.val_check_interval,
        log_every_n_steps=cfg.training.log_every_n_steps,
        logger=loggers,
        callbacks=callbacks
    )

    trainer.fit(model, datamodule=data_module)
    torch.onnx.export(model=model,
                      args=torch.zeros((1, cfg.model.input_dim)),
                      f="models/model.onnx")


if __name__ == "__main__":
    train()
