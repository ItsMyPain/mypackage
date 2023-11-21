from classifier import Classifier


def train(iterations=1000, learning_rate=0.01, depth=6, silent=True):
    model = Classifier(iterations, learning_rate, depth)
    model.train("dataset.csv", "Dx:Cancer", silent)
    model.save()


if __name__ == "__main__":
    train(silent=True)
