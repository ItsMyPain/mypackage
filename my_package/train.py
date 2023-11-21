import requests

from classifier import Classifier

try:
    f = open("dataset.csv")
    f.close()
except FileNotFoundError:
    file_id = "1HJTDfP6njwgsToIrSSPBNNRXvh9QmX_R"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    r = requests.get(url)
    while r.status_code != 200:
        r = requests.get(url)
    with open("dataset.csv", "wb") as f:
        f.write(r.content)

print("File downloaded")


def train(iterations=1000, learning_rate=0.01, depth=6, silent=True):
    model = Classifier(iterations, learning_rate, depth)
    model.train("dataset.csv", "Dx:Cancer", silent)
    model.save()


if __name__ == '__main__':
    train(silent=False)
