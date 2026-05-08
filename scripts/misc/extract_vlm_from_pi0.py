"""
Extract the VLM (Emu3MoE world model) from a trained Emu3Pi0 checkpoint.

The saved Emu3Pi0 checkpoint contains both:
  - vlm.*  keys  → the Emu3MoE world model
  - action_expert.*, action_projector.*, etc. → the action expert

This script filters out vlm.* keys, strips the prefix, and saves a
standalone Emu3MoE checkpoint that can be used as pretrain_vlm_path
for any subsequent MoE / flow-matching training.
"""
import os
import sys
import json
import shutil

import torch
from safetensors.torch import load_file, save_file

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.expanduser("~/Project/DriveVLA-W0")

SRC_DIR  = os.path.join(PROJECT_ROOT, "pretrained_models",
                         "Emu3_Flow_Matching_Action_Expert_PDMS_87.2")
DEST_DIR = os.path.join(PROJECT_ROOT, "pretrained_models",
                         "Emu3_FlowMatching_VLM_only")

os.makedirs(DEST_DIR, exist_ok=True)
print(f"Source : {SRC_DIR}")
print(f"Dest   : {DEST_DIR}")

# ---------------------------------------------------------------------------
# Step 1: Collect safetensors shards and index
# ---------------------------------------------------------------------------
index_path = os.path.join(SRC_DIR, "model.safetensors.index.json")
with open(index_path) as f:
    index = json.load(f)

weight_map = index["weight_map"]          # key → shard filename

# All shards that contain vlm.* keys
vlm_shards = sorted({v for k, v in weight_map.items() if k.startswith("vlm.")})
print(f"\nShards containing VLM weights: {vlm_shards}")

# ---------------------------------------------------------------------------
# Step 2: Extract VLM tensors from each shard
# ---------------------------------------------------------------------------
vlm_tensors   = {}   # stripped_key → tensor
new_weight_map = {}  # stripped_key → new shard filename (we'll use same shard idx)

for shard_fname in vlm_shards:
    shard_path = os.path.join(SRC_DIR, shard_fname)
    print(f"\nLoading {shard_fname} …", flush=True)
    shard = load_file(shard_path)

    for k, v in shard.items():
        if k.startswith("vlm."):
            stripped = k[len("vlm."):]   # e.g. "model.layers.0…"
            vlm_tensors[stripped] = v
            new_weight_map[stripped] = shard_fname   # keep shard assignment

print(f"\nTotal VLM parameter tensors: {len(vlm_tensors)}")

# ---------------------------------------------------------------------------
# Step 3: Save VLM tensors split by original shard
# ---------------------------------------------------------------------------
shard_groups: dict[str, dict] = {}
for key, shard_fname in new_weight_map.items():
    shard_groups.setdefault(shard_fname, {})[key] = vlm_tensors[key]

saved_shards = []
for orig_fname, tensors in shard_groups.items():
    out_path = os.path.join(DEST_DIR, orig_fname)
    print(f"Saving {orig_fname}  ({len(tensors)} tensors) …", flush=True)
    save_file(tensors, out_path, metadata={"format": "pt"})
    saved_shards.append(orig_fname)

# ---------------------------------------------------------------------------
# Step 4: Write new index.json
# ---------------------------------------------------------------------------
new_index = {
    "metadata": {"total_size": sum(t.numel() * t.element_size()
                                   for t in vlm_tensors.values())},
    "weight_map": new_weight_map,
}
new_index_path = os.path.join(DEST_DIR, "model.safetensors.index.json")
with open(new_index_path, "w") as f:
    json.dump(new_index, f, indent=2)
print(f"\nSaved index: {new_index_path}")

# ---------------------------------------------------------------------------
# Step 5: Build a config.json for the extracted Emu3MoE
# ---------------------------------------------------------------------------
src_cfg_path = os.path.join(SRC_DIR, "config.json")
with open(src_cfg_path) as f:
    pi0_cfg = json.load(f)

# vlm_config contains the Emu3MoE architecture parameters
vlm_cfg = pi0_cfg.get("vlm_config", {})

# Merge top-level backbone fields that Emu3MoE expects
moe_cfg = {
    "architectures":         ["Emu3MoE"],
    "model_type":            "Emu3",
    "hidden_size":           pi0_cfg.get("hidden_size",           4096),
    "intermediate_size":     pi0_cfg.get("intermediate_size",     14336),
    "num_hidden_layers":     pi0_cfg.get("num_hidden_layers",     32),
    "num_attention_heads":   pi0_cfg.get("num_attention_heads",   32),
    "num_key_value_heads":   pi0_cfg.get("num_key_value_heads",   8),
    "hidden_act":            pi0_cfg.get("hidden_act",            "silu"),
    "max_position_embeddings": pi0_cfg.get("max_position_embeddings", 1400),
    "rms_norm_eps":          pi0_cfg.get("rms_norm_eps",          1e-5),
    "rope_theta":            pi0_cfg.get("rope_theta",            1000000.0),
    "attention_dropout":     pi0_cfg.get("attention_dropout",     0.0),
    "vocab_size":            pi0_cfg.get("vocab_size",            184622),
    "image_area":            pi0_cfg.get("image_area",            262144),
    "tie_word_embeddings":   pi0_cfg.get("tie_word_embeddings",   False),
    "torch_dtype":           pi0_cfg.get("torch_dtype",           "bfloat16"),
    "transformers_version":  pi0_cfg.get("transformers_version",  "4.44.0"),
    "use_cache":             True,
    "pretraining_tp":        1,
    "pad_token_id":          pi0_cfg.get("pad_token_id",          151643),
    "bos_token_id":          pi0_cfg.get("bos_token_id",          151849),
    "eos_token_id":          pi0_cfg.get("eos_token_id",          151850),
    "boi_token_id":          pi0_cfg.get("boi_token_id",          151852),
    "eoi_token_id":          pi0_cfg.get("eoi_token_id",          151853),
    "bov_token_id":          pi0_cfg.get("bov_token_id",          151854),
    "eov_token_id":          pi0_cfg.get("eov_token_id",          184621),
    "eof_token_id":          pi0_cfg.get("eof_token_id",          151847),
    "eol_token_id":          pi0_cfg.get("eol_token_id",          151846),
    "boa_token_id":          pi0_cfg.get("boa_token_id",          151844),
    "img_token_id":          pi0_cfg.get("img_token_id",          151851),
    "vision_loss_weight":    vlm_cfg.get("vision_loss_weight",    0.5),
    "action_experts":        False,
}

dest_cfg_path = os.path.join(DEST_DIR, "config.json")
with open(dest_cfg_path, "w") as f:
    json.dump(moe_cfg, f, indent=2)
print(f"Saved config: {dest_cfg_path}")

# ---------------------------------------------------------------------------
# Step 6: Copy tokenizer files
# ---------------------------------------------------------------------------
tokenizer_files = [
    "emu3.tiktoken",
    "emu3_vision_tokens.txt",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "generation_config.json",
]
for fname in tokenizer_files:
    src = os.path.join(SRC_DIR, fname)
    dst = os.path.join(DEST_DIR, fname)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"Copied {fname}")

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
print(f"\nDone. Extracted VLM saved to:\n  {DEST_DIR}")
print("\nDirectory contents:")
for f in sorted(os.listdir(DEST_DIR)):
    size = os.path.getsize(os.path.join(DEST_DIR, f))
    print(f"  {f:50s}  {size/1024**2:8.1f} MB")
