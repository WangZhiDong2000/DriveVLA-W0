"""
VQ tokenizer for navsim mini dataset (single GPU version).
Reads images from mini_sensor_blobs and saves VQ codes as .npy files.
"""
import os
import os.path as osp
import torch
from transformers import AutoModel, AutoImageProcessor
import numpy as np
import re
import time
from tqdm import tqdm
from PIL import Image

DATASET_ROOT = os.path.expanduser("~/Dataset/navsim/download")
PROJECT_ROOT = os.path.expanduser("~/Project/DriveVLA-W0")

MODEL_PATH = osp.join(PROJECT_ROOT, "pretrained_models/Emu3-VisionTokenizer")
VIDEO_ROOT = osp.join(DATASET_ROOT, "mini_sensor_blobs/mini")
VIDEO_CODES_SAVE = osp.join(DATASET_ROOT, "processed_data/mini_vq_codes_256_144")
SIZE = (256, 144)


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def load_images(folder_path, size):
    image_paths = sorted(os.listdir(folder_path), key=natural_sort_key)
    images = [Image.open(osp.join(folder_path, img)).resize(size) for img in image_paths]
    return images, image_paths


def image_level_encode(images, image_paths, model, processor, save_codes_path, batch_size=8):
    os.makedirs(save_codes_path, exist_ok=True)
    images_tensor = processor(images, return_tensors="pt")["pixel_values"].cuda()
    num_images = images_tensor.shape[0]
    for start_idx in range(0, num_images, batch_size):
        batch = images_tensor[start_idx:start_idx + batch_size]
        try:
            with torch.no_grad():
                codes = model.encode(batch)
            for idx, code in enumerate(codes):
                img_name = image_paths[start_idx + idx].replace(".jpg", "").replace(".png", "").replace(".jpeg", "")
                np.save(osp.join(save_codes_path, f"{img_name}.npy"), code.detach().cpu().numpy())
        except Exception as e:
            print(f"Error at batch {start_idx}: {e}")


if __name__ == "__main__":
    os.makedirs(VIDEO_CODES_SAVE, exist_ok=True)

    print(f"Loading model from {MODEL_PATH}")
    model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True).eval().cuda()
    processor = AutoImageProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    processor.min_pixels = SIZE[0] * SIZE[1]

    all_logs = sorted(os.listdir(VIDEO_ROOT))
    print(f"Found {len(all_logs)} logs to process")

    for log_name in tqdm(all_logs, desc="Processing logs"):
        cam_f0_path = osp.join(VIDEO_ROOT, log_name, "CAM_F0")
        if not osp.isdir(cam_f0_path):
            print(f"Skipping {log_name}: no CAM_F0 folder")
            continue

        save_path = osp.join(VIDEO_CODES_SAVE, log_name)
        if osp.exists(save_path) and len(os.listdir(save_path)) > 0:
            print(f"Skipping {log_name}: already processed ({len(os.listdir(save_path))} files)")
            continue

        print(f"Processing {log_name}")
        t0 = time.time()
        images, image_paths = load_images(cam_f0_path, SIZE)
        image_level_encode(images, image_paths, model, processor, save_path)
        print(f"  Done in {time.time() - t0:.1f}s, saved {len(image_paths)} codes")
