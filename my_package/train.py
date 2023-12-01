import hydra
import lightning.pytorch as pl
import lightning.pytorch.loggers as pl_loggers
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from dvc.api import DVCFileSystem
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchmetrics.classification import F1Score

from my_package.classes import CustomDataset, Classifier
from my_package.data import MyDataModule
from my_package.model import MyModel


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(cfg: DictConfig):
    fs = DVCFileSystem()
    fs.get_file(f"data/{cfg.train.dataset}", f"data/{cfg.train.dataset}")

    batch_size = cfg.train.batch_size
    epochs = cfg.train.epochs
    lr = cfg.train.lr

    full_data = pd.read_csv(f"data/{cfg.train.dataset}")
    X = full_data.drop(columns=cfg.train.target)
    y = full_data[cfg.train.target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=cfg.train.random_state,
        train_size=cfg.train.train_size, stratify=y
    )
    X_train, y_train = np.array(X_train, dtype=np.float32), y_train.values
    X_test, y_test = np.array(X_test, dtype=np.float32), y_test.values

    num_features = X.shape[-1]
    num_classes = np.unique(y).shape[0]

    train_dataset = CustomDataset(
        torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long()
    )
    valid_dataset = CustomDataset(
        torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long()
    )

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Classifier(num_features, num_classes).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    train_losses = []
    train_accs = []
    valid_losses = []
    valid_accs = []
    train_f1s = []
    valid_f1s = []

    best_valid_acc = 0.0

    f1 = F1Score(task="binary", num_classes=2)
    best_model = model

    for epoch in range(epochs):
        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0
        train_f1 = 0.0
        valid_f1 = 0.0

        model.train()
        for data in train_loader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            total = labels.size(0)
            acc = correct / total
            train_acc += acc
            f1_m = f1(predicted, labels).item()
            train_f1 += f1_m

        model.eval()
        with torch.no_grad():
            for data in valid_loader:
                inputs, labels = data

                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                valid_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == labels).sum().item()
                total = labels.size(0)
                acc = correct / total
                valid_acc += acc
                f1_m = f1(predicted, labels).item()
                valid_f1 += f1_m

        train_loss = train_loss / len(train_loader)
        train_acc = train_acc / len(train_loader)
        train_f1 = train_f1 / len(train_loader)
        valid_loss = valid_loss / len(valid_loader)
        valid_acc = valid_acc / len(valid_loader)
        valid_f1 = valid_f1 / len(valid_loader)

        print(
            "Epoch: %d | Train Loss: %.3f | Train Acc: %.3f | "
            "Valid Loss: %.3f | Valid Acc: %.3f"
            % (epoch + 1, train_loss, train_acc, valid_loss, valid_acc)
        )
        print(train_f1, valid_f1)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)
        train_f1s.append(train_f1)
        valid_f1s.append(valid_f1)

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            print("Best Valid Acc Improved: %.3f" % best_valid_acc)
            best_model = model
    best_model.save()
    print("Finished Training")


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def train_v2(cfg: DictConfig):
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
            tracking_uri=cfg.mlflow.uri
        )
    ]

    callbacks = [
        pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
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


if __name__ == "__main__":
    # train()
    train_v2()
