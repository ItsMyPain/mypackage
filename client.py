import hydra
import numpy as np
import pandas as pd
from dvc.fs import DVCFileSystem
from functools import lru_cache
from omegaconf import DictConfig
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput
from tritonclient.utils import np_to_triton_dtype


@lru_cache
def get_client():
    return InferenceServerClient(url="0.0.0.0:8500")


def call_triton(data):
    triton_client = get_client()

    input_text = InferInput(
        name="INPUT", shape=data.shape, datatype=np_to_triton_dtype(data.dtype)
    )
    input_text.set_data_from_numpy(data, binary_data=True)

    query_response = triton_client.infer(
        "onnx-model",
        [input_text],
        outputs=[
            InferRequestedOutput("PROBS", binary_data=True),
        ],
    )

    probs = query_response.as_numpy("PROBS")[0]
    return probs


@hydra.main(version_base=None, config_path="configs", config_name="test")
def main(cfg: DictConfig):
    fs = DVCFileSystem()
    fs.get_file(cfg.data.name, cfg.data.name)

    df = pd.read_csv(cfg.data.name).head(1)
    inputs = df.drop(columns=cfg.data.target).values
    target = df[cfg.data.target].values

    outputs = [call_triton(input) for input in inputs]

    print(outputs)

    assert len(outputs) == len(inputs)
    assert (target == outputs).all()


if __name__ == "__main__":
    main()
