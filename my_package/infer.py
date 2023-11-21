from my_package.classifier import Classifier


def infer():
    model = Classifier()
    model.load()
    model.predict("dataset.csv", "Dx:Cancer")


if __name__ == "__main__":
    infer()
