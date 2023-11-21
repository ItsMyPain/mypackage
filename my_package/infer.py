import requests

from mypackage.classifier import Classifier

try:
    open("dataset.csv")
except FileNotFoundError:
    file_id = "1HJTDfP6njwgsToIrSSPBNNRXvh9QmX_R"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    r = requests.get(url)
    while r.status_code != 200:
        r = requests.get(url)
    with open("dataset.csv", "wb") as f:
        f.write(r.content)


def infer():
    model = Classifier()
    model.load()
    model.predict("dataset.csv", "Dx:Cancer")


if __name__ == '__main__':
    infer()