import json
import random
import warnings
warnings.filterwarnings("ignore")

import os
import os.path as osp
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Optional, List
import pathlib
import transformers as tf
from datasets import Emu3SFTDataset
from torch.utils.data.dataloader import default_collate
import sys
# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取父目录(即包含train和reference的目录)
parent_dir = os.path.dirname(current_dir)
# 添加reference/Emu3路径到sys.path
sys.path.append(os.path.join(parent_dir, "reference", "Emu3"))
from emu3.mllm import Emu3Config, Emu3Tokenizer, Emu3ForCausalLM, Emu3MoE, Emu3MoEConfig, Emu3Pi0, Emu3Pi0Config
from transformers import AutoModel,Trainer
from datasets import Emu3DrivingDataset
from datasets import Emu3DrivingVAVADataset
from torch.utils.data import WeightedRandomSampler, DataLoader

class WeightedSamplerTrainer(Trainer):
    def get_train_dataloader(self):
        # 从 train_dataset 中获取 sample_weights
        sample_weights = torch.tensor(
            self.train_dataset.sample_weights, dtype=torch.double
        )
        # 用 sample_weights 构建 WeightedRandomSampler
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="BAAI/Emu3-Gen")
    model_config_path: Optional[str] = field(default="pretrain/Emu3-Base")
    pretrain_vlm_path: Optional[str] = field(default=None)
    init_fresh_expert: bool = field(default=False)

@dataclass
class DataArguments:
    data_path: Optional[str] = field(default=None)
    null_prompt_prob: float = field(default=0.05)
    apply_loss_on_only_vision: bool = field(default=True)
    apply_loss_on_only_text: bool = field(default=False)
    apply_loss_on_only_action: bool = field(default=False) 
    ignore_index: int = field(default=-100)
    visual_token_pattern: str = field(default="<|visual token {token_id:0>6d}|>")
    codebook_size: Optional[int] = field(default=32768)
    frames: int = field(default=4)
    VL: bool = field(default=False)
    actions: bool = field(default=False)
    actions_format: str = field(default="openvla")
    action_frames: int = field(default=8)
    use_gripper: bool = field(default=False)
    action_tokenizer_path: Optional[str] = field(default=None)
    video_format: str = field(default=None)
    random_frame_sampling: bool = field(default=True)
    raw_image: bool = field(default=False)
    post_training: bool = field(default=False)
    datasets_weight: bool = field(default=False)
    without_text: bool = field(default=False)
    real_robot: bool = field(default=False)
    driving: bool = field(default=False)
    use_previous_actions: bool = field(default=False)
    use_only_lidar: bool = field(default=False)
    use_lidar_and_image: bool = field(default=False)
    use_flip: bool = field(default=False)
    cur_frame_idx: int = field(default=3)
    action_dim: int = field(default=3)  # Action dimension for Pi0 model
    pre_action_frames: int = field(default=3)
    normalizer_path: Optional[str] = field(default=None)

@dataclass
class TrainingArguments(tf.TrainingArguments):
    report_to: List[str] = field(default_factory=list)
    remove_unused_columns: bool = field(default=False)
    min_learning_rate: Optional[float] = field(default=None)
    attn_type: Optional[str] = field(default="fa2")
    image_area: Optional[int] = field(default=None)
    max_position_embeddings: Optional[int] = field(default=None)
    from_scratch: bool = field(default=False)
    dataloader_num_workers: Optional[int] = field(default=0)
    evaluation_strategy: str = field(default="steps")  # or "epoch"
    eval_steps: Optional[int] = field(default=1000)     # 每 1000 step 验证一次
    per_device_eval_batch_size: Optional[int] = field(default=1)
    eval_accumulation_steps: Optional[int] = field(default=1)
    # Pi0 specific training arguments
    train_action_only: bool = field(default=False)
    action_loss_weight: float = field(default=10.0)
    action_sample_steps: int = field(default=10)
    freeze_vlm: bool = field(default=False)  # 新增：是否冻结VLM参数

def _load_vlm_from_pi0_checkpoint(model, checkpoint_path):
    """Load only VLM weights from a Pi0 checkpoint, shard by shard, to minimise peak RAM."""
    from safetensors import safe_open
    import glob
    import gc

    shard_files = sorted(glob.glob(osp.join(checkpoint_path, "model-*.safetensors")))
    if not shard_files:
        shard_files = sorted(glob.glob(osp.join(checkpoint_path, "*.safetensors")))

    print(f"Loading VLM weights from {len(shard_files)} Pi0 shards in {checkpoint_path}")
    for shard_path in shard_files:
        with safe_open(shard_path, framework='pt', device='cpu') as f:
            vlm_keys = [k for k in f.keys() if k.startswith("vlm.")]
            if not vlm_keys:
                continue
            vlm_dict = {k[4:]: f.get_tensor(k) for k in vlm_keys}  # strip "vlm." prefix
        model.vlm.load_state_dict(vlm_dict, strict=False)
        print(f"  {len(vlm_dict)} VLM keys from {osp.basename(shard_path)}")
        del vlm_dict
        gc.collect()
    print("VLM weights loaded from Pi0 checkpoint. Action expert remains freshly initialized.")


def load_model(model_args, model_config, training_args):
    with open(osp.join(model_args.model_name_or_path, "config.json"), "r") as f:
        config = json.load(f)

    if config.get("model_type") == "Emu3Pi0" and model_args.init_fresh_expert:
        # Load trained VLM from Pi0 checkpoint; action expert is freshly initialized.
        model = Emu3Pi0(config=model_config, pretrain_vlm_path=None)
        _load_vlm_from_pi0_checkpoint(model, model_args.model_name_or_path)
        if training_args.freeze_vlm:
            print("Freezing VLM parameters...")
            model.freeze_vlm()
    elif config.get("model_type") == "Emu3Pi0":
        # 直接读取这个模型
        model_config = Emu3Pi0Config.from_pretrained(os.path.join(model_args.model_name_or_path, "config.json"))
        model, loading_info = Emu3Pi0.from_pretrained(
            model_args.model_name_or_path,
            config=model_config,
            pretrain_vlm_path=model_args.pretrain_vlm_path or model_args.model_name_or_path,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            output_loading_info=True
        )
        print("Missing keys in Emu3Pi0 model:", loading_info["missing_keys"])
        print("Unexpected keys in Emu3Pi0 model:", loading_info["unexpected_keys"])
        print("Mismatched sizes in Emu3Pi0 model:", loading_info.get("mismatched_keys", "N/A"))
    else:
        # 初始化 Pi0 模型
        model = Emu3Pi0(config=model_config, pretrain_vlm_path=model_args.model_name_or_path)
        if training_args.freeze_vlm:
            print("Freezing VLM parameters...")
            model.freeze_vlm()

    return model


def get_dataset(data_args, tokenizer):
    """
    Initialize and return the training dataset.
    """
    if data_args.driving:
        return Emu3DrivingVAVADataset(data_args, tokenizer=tokenizer)
    return Emu3SFTDataset(data_args, tokenizer=tokenizer)

def get_dataset_split(data_args, tokenizer):
    if data_args.post_training:
        full_dataset = Emu3WorldModelDataset(data_args, tokenizer=tokenizer)
    elif data_args.driving:
        full_dataset = Emu3DrivingVAVADataset(data_args, tokenizer=tokenizer)
    else:
        full_dataset = Emu3SFTDataset(data_args, tokenizer=tokenizer)
    split = full_dataset.train_test_split(test_size=0.05, seed=42)
    return split["train"], split["test"]

def update_configs(model_config, args, fields):
    cross_update = lambda a, b, field_name: (
        setattr(b, field_name, getattr(a, field_name))
        if getattr(b, field_name, None) is None else
        setattr(a, field_name, getattr(b, field_name))
    )

    for f in fields:
        cross_update(model_config, args, f)

class L1EvalCallback(tf.TrainerCallback):
    """Compute trajectory L1 at 1s/2s/3s after each HF Trainer eval, log to WandB."""
    # action_frames=8 at 0.5s each → 1s=idx1, 2s=idx3, 3s=idx5
    HORIZON_INDICES = {1: 1, 2: 3, 3: 5}

    def __init__(self, eval_dataset, normalizer_path,
                 num_eval_samples=64, eval_batch_size=2, action_sample_steps=5):
        self.eval_dataset = eval_dataset
        self.num_eval_samples = num_eval_samples
        self.eval_batch_size = eval_batch_size
        self.action_sample_steps = action_sample_steps
        self.q01 = self.q99 = None
        if normalizer_path:
            q01 = np.load(osp.join(normalizer_path, "libero_q01.npy"))
            q99 = np.load(osp.join(normalizer_path, "libero_q99.npy"))
            self.q01 = torch.tensor(q01, dtype=torch.float32)  # [3]
            self.q99 = torch.tensor(q99, dtype=torch.float32)  # [3]

    def _denorm(self, x, device):
        """Denormalize from [-1,1] to physical units (m, m, rad)."""
        if self.q01 is None:
            return None
        return (x + 1) / 2 * (self.q99.to(device) - self.q01.to(device)) + self.q01.to(device)

    def on_evaluate(self, args, state, control, model, **kwargs):
        try:
            import wandb
        except ImportError:
            return
        if wandb.run is None:
            return

        n = min(self.num_eval_samples, len(self.eval_dataset))
        indices = list(range(len(self.eval_dataset)))
        random.shuffle(indices)
        indices = indices[:n]

        l1_norm = {1: [], 2: [], 3: []}
        l1_phys = {1: [], 2: [], 3: []}
        was_training = model.training

        for start in range(0, n, self.eval_batch_size):
            batch_idx = indices[start:start + self.eval_batch_size]
            batch = default_collate([self.eval_dataset[i] for i in batch_idx])
            device = next(model.parameters()).device
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pre_action     = batch["pre_action"].to(device)
            cmd            = batch["cmd"].to(device)
            gt_action      = batch["action"].to(device)  # [B, 8, 3]

            with torch.no_grad():
                pred = model.sample_actions(
                    input_ids=input_ids, pre_action=pre_action, cmd=cmd,
                    attention_mask=attention_mask,
                    num_inference_steps=self.action_sample_steps,
                )  # [B, 8, 3]

            for horizon, idx in self.HORIZON_INDICES.items():
                l1 = torch.abs(pred[:, idx] - gt_action[:, idx]).mean().item()
                l1_norm[horizon].append(l1)
                phys_pred = self._denorm(pred[:, idx], device)
                phys_gt   = self._denorm(gt_action[:, idx], device)
                if phys_pred is not None:
                    l1_phys[horizon].append(torch.abs(phys_pred - phys_gt).mean().item())

        if was_training:
            model.train()

        metrics = {f"eval_l1/{h}s_norm": float(np.mean(v)) for h, v in l1_norm.items()}
        if l1_phys[1]:
            metrics.update({f"eval_l1/{h}s_phys": float(np.mean(v)) for h, v in l1_phys.items()})
        wandb.log(metrics)  # no explicit step: avoids out-of-order warning with HF Trainer


def train():
    """
    Main function to train the model.
    """
    # Parse arguments
    parser = tf.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set environment variable for WANDB logging
    os.environ["WANDB_DIR"] = osp.join(training_args.output_dir, "wandb")

    # Load Pi0 configuration
    pi0_config = Emu3Pi0Config.from_pretrained(model_args.model_config_path)
    update_configs(pi0_config, training_args, ["image_area", "max_position_embeddings", "action_loss_weight", "action_sample_steps", "freeze_vlm"])
    if training_args.bf16:
        pi0_config.torch_dtype = torch.bfloat16
        pi0_config.vlm_config.torch_dtype = torch.bfloat16
        pi0_config.action_config.torch_dtype = torch.bfloat16
    
    # Initialize model
    model = load_model(model_args, pi0_config, training_args)

    if training_args.min_learning_rate is not None:
        training_args.lr_scheduler_kwargs["min_lr"] = training_args.min_learning_rate
    
    tokenizer = Emu3Tokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.max_position_embeddings,
        padding_side="right",
        use_fast=False,
    )

    # Initialize dataset
    train_dataset, eval_dataset = get_dataset_split(data_args, tokenizer)

    # Build L1 eval callback if WandB is enabled and normalizer path is provided
    callbacks = []
    if data_args.normalizer_path and "wandb" in training_args.report_to:
        callbacks.append(L1EvalCallback(
            eval_dataset=eval_dataset,
            normalizer_path=data_args.normalizer_path,
            num_eval_samples=64,
            eval_batch_size=2,
            action_sample_steps=training_args.action_sample_steps,
        ))

    if data_args.datasets_weight:
        trainer = WeightedSamplerTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            callbacks=callbacks or None,
        )
    else:
        trainer = tf.Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            callbacks=callbacks or None,
        )


    # Check if resuming from checkpoint
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save model and training state
    trainer.save_state()
    torch.cuda.synchronize()
    trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    train()
