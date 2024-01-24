#!/usr/bin/python3

import os
import sys
from urllib import request


MODELS_URLS = {
    "zephyr1_6b": {
        "fp16": "https://huggingface.co/iangitonga/gten/resolve/main/zephyr1_6b.fp16.gten",
        "q8": "https://huggingface.co/iangitonga/gten/resolve/main/zephyr1_6b.q8.gten",
        "q4": "https://huggingface.co/iangitonga/gten/resolve/main/zephyr1_6b.q4.gten",
    }
}


def _download_model(url, model_path):
    print("Downloading model ...")
    with request.urlopen(url) as source, open(model_path, "wb") as output:
        download_size = int(source.info().get("Content-Length"))
        download_size_mb = int(download_size / 1000_000)
        downloaded = 0
        while True:
            buffer = source.read(8192)
            if not buffer:
                break

            output.write(buffer)
            downloaded += len(buffer)
            progress_perc = int(downloaded / download_size * 100.0)
            downloaded_mb = int(downloaded / 1000_000)
            # trick to make it work with jupyter.
            print(f"\rDownload Progress [{downloaded_mb}MB/{download_size_mb}MB]: {progress_perc}%", end="")
    print("\n\n")

def download_model(dtype):
    model_name = "zephyr1_6b"
    model_path = os.path.join("models", f"{model_name}.{dtype}.gten")

    if os.path.exists(model_path):
        return
    os.makedirs("models", exist_ok=True)
    model_url_key = f"{model_name}.{dtype}"
    _download_model(MODELS_URLS[model_name][dtype], model_path)

# python model.py model inf
if len(sys.argv) != 2:
    print(f"Args provided: {sys.argv}")
    print("DTYPE is one of (fp16, q8, q4)")
    exit(-1)


try:
    download_model(sys.argv[1])
except:
    exit(-2)
