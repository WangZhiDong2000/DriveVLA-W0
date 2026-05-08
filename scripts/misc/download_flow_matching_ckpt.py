"""
Download Emu3_Flow_Matching_Action_Expert_PDMS_87.2 from liyingyan/DriveVLA-W0.
Saves to pretrained_models/Emu3_Flow_Matching_Action_Expert_PDMS_87.2/
"""
import os
import sys

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import hf_hub_download

REPO_ID = "liyingyan/DriveVLA-W0"
FOLDER = "Emu3_Flow_Matching_Action_Expert_PDMS_87.2"
LOCAL_DIR = os.path.expanduser("~/Project/DriveVLA-W0/pretrained_models")

FILES = [
    f"{FOLDER}/config.json",
    f"{FOLDER}/generation_config.json",
    f"{FOLDER}/special_tokens_map.json",
    f"{FOLDER}/tokenizer_config.json",
    f"{FOLDER}/emu3.tiktoken",
    f"{FOLDER}/emu3_vision_tokens.txt",
    f"{FOLDER}/model.safetensors.index.json",
    f"{FOLDER}/trainer_state.json",
    f"{FOLDER}/training_args.bin",
    f"{FOLDER}/model-00001-of-00004.safetensors",
    f"{FOLDER}/model-00002-of-00004.safetensors",
    f"{FOLDER}/model-00003-of-00004.safetensors",
    f"{FOLDER}/model-00004-of-00004.safetensors",
]

print(f"Downloading {len(FILES)} files to {LOCAL_DIR}/{FOLDER}/")
print()

for i, filename in enumerate(FILES, 1):
    basename = filename.split("/")[-1]
    dest = os.path.join(LOCAL_DIR, FOLDER, basename)
    if os.path.exists(dest):
        size_mb = os.path.getsize(dest) / 1024**2
        print(f"[{i}/{len(FILES)}] SKIP (exists, {size_mb:.0f} MB): {basename}")
        sys.stdout.flush()
        continue
    print(f"[{i}/{len(FILES)}] Downloading: {basename} ...", flush=True)
    hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        local_dir=LOCAL_DIR,
        local_dir_use_symlinks=False,
    )
    size_mb = os.path.getsize(dest) / 1024**2
    print(f"[{i}/{len(FILES)}] Done: {basename} ({size_mb:.0f} MB)", flush=True)

print()
print("All files downloaded successfully.")
print(f"Model saved to: {LOCAL_DIR}/{FOLDER}/")
