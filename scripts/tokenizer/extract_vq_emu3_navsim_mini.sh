#!/bin/bash
# VQ code generation for navsim mini dataset (single GPU)
# Processes all 64 mini logs on GPU 0
CUDA_VISIBLE_DEVICES=0 python3 models/tokenizer/emu3_tokenizer_navsim_mini.py
