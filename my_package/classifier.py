from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from imblearn.over_sampling import SMOTE
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


class Classifier:
    model: CatBoostClassifier
    scaler: MinMaxScaler
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.DataFrame
    y_test: pd.DataFrame

    def __init__(self, iterations=None, learning_rate=None, depth=None):
        self.model = CatBoostClassifier(iterations, learning_rate, depth)
        self.scaler = MinMaxScaler()
        self.smote = SMOTE()

    def _smote_data(self):
        data = self.smote.fit_resample(self.X_train, self.y_train)
        self.X_train, self.y_train = data

    def _scale_data(self, x):
        return self.scaler.fit_transform(x)

    def _prepare_data(self, dataset: pd.DataFrame, target: str):
        dataset = dataset.drop_duplicates()
        target = target
        df2 = dataset.loc[:, :]

        for name in dataset.columns:
            mean = np.array(dataset[f"{name}"] != "?").mean()
            df2[name] = dataset[name].replace({"?": mean})

        x = df2.drop(columns=target)
        y = df2[target]

        datasets = train_test_split(x, y, test_size=0.33, stratify=y)
        self.X_train, self.X_test, self.y_train, self.y_test = datasets

    def train(self, filename: str, target: str, silent=True):
        df = pd.read_csv(f"data/{filename}")
        self._prepare_data(df, target)
        self._smote_data()
        self.X_train = self._scale_data(self.X_train)
        train_pool = Pool(self.X_train, self.y_train)
        self.model.fit(train_pool, silent=silent)
        print("Model fitted")

    def predict(self, filename: str, target: str):
        df = pd.read_csv(f"data/{filename}")
        self._prepare_data(df, target)
        self.X_test = self._scale_data(self.X_test)
        y_pred = self.model.predict(self.X_test)
        score = recall_score(self.y_test, y_pred, average="macro")
        print("RECALL SCORE: ", score)
        ans = pd.DataFrame({"predict": y_pred})
        ans.to_csv("predict.csv")

    def save(self, directory="classifier", model="model", scaler="scaler"):
        Path(f"models/{directory}").mkdir(parents=True, exist_ok=True)
        self.model.save_model(f"models/{directory}/{model}")
        print("Model saved")

        joblib.dump(self.scaler, f"models/{directory}/{scaler}")
        print("Scaler saved")

    def load(self, directory="classifier", model="model", scaler="scaler"):
        self.model.load_model(f"models/{directory}/{model}")
        print("Model loaded")
        self.scaler = joblib.load(f"models/{directory}/{scaler}")
        print("Scaler loaded")
