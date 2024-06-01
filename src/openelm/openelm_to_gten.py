import argparse

import torch
import numpy as np
from safetensors import safe_open


GTEN_MAGIC_NUMBER = 0x454c49464e455447
BYTEORDER = "little"

# unsigned int to bytes.
def itob(integer, width=4):
    return int.to_bytes(integer, width, BYTEORDER, signed=True)

# float to bytes
def ftob(floatv):
    return np.array([floatv]).astype(np.float32).tobytes()



def write_layer(fout, name: str, w0: torch.Tensor, dtype: str):
    name = name.encode()
    # <layer_name_size, layer_name>
    fout.write(itob(len(name)))
    fout.write(name)

    w0_name = name
    fout.write(itob(len(w0_name)))
    fout.write(w0_name)

    if dtype == "fp16":
        w0 = w0.to(torch.float16)
        w0 = w0.cpu().numpy().flatten()
        w0_bytes = w0.tobytes()
        fout.write(itob(len(w0_bytes)))
        fout.write(w0_bytes)
    else:
        assert(False)

"""

transformer.layers.0.attn.k_norm.weight
transformer.layers.0.attn.out_proj.weight
transformer.layers.0.attn.q_norm.weight
transformer.layers.0.attn.qkv_proj.weight
transformer.layers.0.attn_norm.weight
transformer.layers.0.ffn.proj_1.weight
transformer.layers.0.ffn.proj_2.weight
transformer.layers.0.ffn_norm.weight

transformer.norm.weight
transformer.token_embeddings.weight

"""

MODELS_CONFIG = {
    # 270M
    "openelm-sm": {
        "n_layers": 16,
        "model_dim": 1280,
        "num_query_heads": [12, 12, 12, 12, 12, 16, 16, 16, 16, 16, 16, 16, 20, 20, 20, 20],
        "num_kv_heads": [3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5],
        "ffn_intermediate_dim": []
    },
    # 450M
    "openelm-md": {
        "n_layers": 20,
        "model_dim": 1536,
        "num_query_heads": [12, 12, 12, 16, 16, 16, 16, 16, 16, 16, 20, 20, 20, 20, 20, 20, 24, 24, 24, 24],
        "num_kv_heads": [3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6],
        "ffn_intermediate_dim": [768, 1024, 1280, 1536, 1792, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4352, 4608, 5120, 5376, 5632, 5888, 6144]
    },
    # 1.1B
    "openelm-lg": {
        "n_layers": 28,
        "model_dim": 2048,
        "num_query_heads": [16, 16, 16, 20, 20, 20, 20, 20, 20, 20, 24, 24, 24, 24, 24, 24, 24, 24, 28, 28, 28, 28, 28, 28, 32, 32, 32, 32],
        "num_kv_heads": [4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8],
        "ffn_intermediate_dim": [1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 4608, 5120, 5376, 5632, 5888, 6144, 6400, 6656, 6912, 7168, 7424, 7680, 7936, 8192]
    },
    # 3B
    "openelm-xl": {
        "n_layers": 36,
        "model_dim": 3072,
        "num_query_heads": [12, 12, 12, 12, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 24, 24, 24, 24, 24, 24],
        "num_kv_heads": [3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6],
        "ffn_intermediate_dim": [1536, 1792, 2048, 2560, 2816, 3072, 3328, 3584, 4096, 4352, 4608, 4864, 5120, 5632, 5888, 6144, 6400, 6656, 7168, 7424, 7680, 7936, 8192, 8704, 8960, 9216, 9472, 9728, 10240, 10496, 10752, 11008, 11264, 11776, 12032, 12288]
    },
}


def convert_model_to_gten(model_path, model_size, dtype):
    ckpt = {}
    with safe_open(model_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            ckpt[k] = f.get_tensor(k)

    model_name = f"openelm-{model_size}"
    out_model_path = f"{model_name}.{dtype}.gten"

    with open(out_model_path, "wb") as fout:
        fout.write(itob(GTEN_MAGIC_NUMBER, width=8))
        
        print("Converting wte")
        name = "transformer.token_embeddings.weight"
        write_layer(fout, name, w0=ckpt[name], dtype=dtype)
        
        n_layers = MODELS_CONFIG[model_name]["n_layers"]
        for i in range(n_layers):
            print(f"Converting block_{i}")

            blk_name = f"transformer.layers.{i}"

            name = f"{blk_name}.attn.qkv_proj.weight"
            d_head = 64
            q_heads = MODELS_CONFIG[model_name]["num_query_heads"][i]
            kv_heads = MODELS_CONFIG[model_name]["num_kv_heads"][i]
            q = q_heads * d_head
            kv = kv_heads * d_head
            query = ckpt[name][:q]
            key = ckpt[name][q : q + kv]
            value = ckpt[name][q + kv : ]

            name = f"{blk_name}.attn.q_proj.weight"
            write_layer(fout, name, w0=query, dtype=dtype)

            name = f"{blk_name}.self_attn.k_proj.weight"
            write_layer(fout, name, w0=key, dtype=dtype)

            name = f"{blk_name}.self_attn.v_proj.weight"
            write_layer(fout, name, w0=value, dtype=dtype)

            name = f"{blk_name}.attn.out_proj.weight"
            write_layer(fout, name, w0=ckpt[name], dtype=dtype)

            name = f"{blk_name}.ffn.proj_1.weight"
            split_idx = ckpt[name].shape[0] // 2
            gate_proj_weight = ckpt[name][:split_idx]
            up_proj_weight = ckpt[name][split_idx:]

            name = f"{blk_name}.mlp.gate_proj.weight"
            write_layer(fout, name, w0=gate_proj_weight, dtype=dtype)

            name = f"{blk_name}.mlp.up_proj.weight"
            write_layer(fout, name, w0=up_proj_weight, dtype=dtype)

            name = f"{blk_name}.mlp.down_proj.weight"
            write_layer(fout, name, w0=ckpt[f"{blk_name}.ffn.proj_2.weight"], dtype=dtype)

            name = f"{blk_name}.attn.q_norm.weight"
            write_layer(fout, name, w0=ckpt[name], dtype="fp16")

            name = f"{blk_name}.attn.k_norm.weight"
            write_layer(fout, name, w0=ckpt[name], dtype="fp16")

            name = f"{blk_name}.attn_norm.weight"
            write_layer(fout, name, w0=ckpt[name], dtype="fp16")

            name = f"{blk_name}.ffn_norm.weight"
            write_layer(fout, name, w0=ckpt[name], dtype="fp16")
        
        print("Converting norm")
        write_layer(fout, "model.norm.weight", w0=ckpt["transformer.norm.weight"], dtype="fp16")


parser = argparse.ArgumentParser()
parser.add_argument("mpath", help="Model path to be converted.")
parser.add_argument("msize", help="size of the model (sm, md, lg, xl)")

args = parser.parse_args()
convert_model_to_gten(args.mpath, args.msize, dtype="fp16")
