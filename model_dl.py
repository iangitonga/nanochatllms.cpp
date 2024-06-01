#!/usr/bin/python3

import os
import sys
from urllib import request


MODELS_URLS = {
    "tinyllama": {
        "fp16": "https://huggingface.co/iangitonga/gten/resolve/main/tinyllama.fp16.gten",
        "q8": "https://huggingface.co/iangitonga/gten/resolve/main/tinyllama.q8.gten",
        "q4": "https://huggingface.co/iangitonga/gten/resolve/main/tinyllama.q4.gten",
    },
    "zephyr": {
        "fp16": "https://huggingface.co/iangitonga/gten/resolve/main/zephyr.fp16.gten",
        "q8": "https://huggingface.co/iangitonga/gten/resolve/main/zephyr.q8.gten",
        "q4": "https://huggingface.co/iangitonga/gten/resolve/main/zephyr.q4.gten",
    },
    "minicpm": {
        "fp16": "https://huggingface.co/iangitonga/gten/resolve/main/minicpm.fp16.gten",
        "q8": "https://huggingface.co/iangitonga/gten/resolve/main/minicpm.q8.gten",
        "q4": "https://huggingface.co/iangitonga/gten/resolve/main/minicpm.q4.gten",
    },
    "openelm_sm": {
        "fp16": "https://huggingface.co/iangitonga/gten/resolve/main/openelm-sm.fp16.gten"
    },
    "openelm_md": {
        "fp16": "https://huggingface.co/iangitonga/gten/resolve/main/openelm-md.fp16.gten"
    },
    "openelm_lg": {
        "fp16": "https://huggingface.co/iangitonga/gten/resolve/main/openelm-lg.fp16.gten"
    },
    "openelm_xl": {
        "fp16": ""
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

def download_model(model_name, dtype):
    model_path = os.path.join("models", f"{model_name}.{dtype}.gten")

    if os.path.exists(model_path):
        return
    os.makedirs("models", exist_ok=True)
    _download_model(MODELS_URLS[model_name][dtype], model_path)

# python model.py model inf
if len(sys.argv) != 3:
    print(f"Args provided: {sys.argv}")
    print("DTYPE is one of (fp16, q8, q4)")
    exit(-1)


try:
    download_model(sys.argv[1], sys.argv[2])
except:
    exit(-2)
