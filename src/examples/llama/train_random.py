"""
Example of how to train a Llama transformer language model.

Launch this with torchrun:

    torchrun --nproc-per-node=4 src/examples/llama/train.py run_name [OVERRIDES...]
"""

import sys
from dataclasses import dataclass
from typing import List, cast

from olmo_core.config import Config, DType
from olmo_core.data import (
    NumpyDataLoaderConfig,
    NumpyDatasetConfig,
    NumpyDatasetType,
    TokenizerConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.nn.transformer import TransformerConfig, TransformerDataParallelConfig
from olmo_core.optim import AdamWConfig, CosWithWarmup, OptimGroupOverride
from olmo_core.train import (
    Duration,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    CometCallback,
    ConfigSaverCallback,
    DownstreamEvaluatorCallbackConfig,
    GPUMemoryMonitorCallback,
    GradClipperCallback,
    LMEvaluatorCallbackConfig,
    ProfilerCallback,
    SchedulerCallback,
    SequenceLengthSchedulerCallback,
    WandBCallback,
)
from olmo_core.utils import get_default_device, seed_all


# modify default checkpointer to not save post-train checkpoint
class NoPostTrainCheckpointer(CheckpointerCallback):
    def post_train(self):
        print("Skipping post-train checkpoint")
        pass

checkpointer_callback = NoPostTrainCheckpointer(save_interval=99999999, pre_train_checkpoint=False, ephemeral_save_interval=9999999, save_async=True)

import os
data_root = "/tmp"
data_root = os.environ.get("HOME") + "/olmo-tmp"


from olmo_core.aliases import PathOrStr
from olmo_core.data import DataCollator, DataLoaderBase


import random
from typing import Any, Dict, Iterable, List, Optional

import torch

class CustomDataLoader(DataLoaderBase):
    """
    An example custom data loader that generates random token IDs.
    """

    def __init__(
        self,
        *,
        sequence_length: int,
        vocab_size: int,
        work_dir: PathOrStr,
        global_batch_size: int,
        dp_world_size: int = 1,
        dp_rank: int = 0,
        fs_local_rank: int = 0,
        seed: int = 0,
        total_batches: int = 2048,
    ):
        super().__init__(
            collator=DataCollator(pad_token_id=vocab_size - 1),
            work_dir=work_dir,
            global_batch_size=global_batch_size,
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
            fs_local_rank=fs_local_rank,
        )
        assert self.rank_batch_size % sequence_length == 0
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.seed = seed
        self._total_batches = total_batches
        self._dataset: Optional[List[torch.Tensor]]

    @property
    def total_batches(self) -> int:
        return self._total_batches

    def state_dict(self) -> Dict[str, Any]:
        return {
            "batches_processed": self.batches_processed,
            "seed": self.seed,
            "epoch": self._epoch,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.batches_processed = state_dict["batches_processed"]
        self.seed = state_dict["seed"]
        self._epoch = state_dict["epoch"]

    def reshuffle(self, epoch: Optional[int] = None, **kwargs):
        del kwargs  # unused

        # Set current epoch.
        if epoch is None:
            epoch = 1 if self._epoch is None else self._epoch + 1
        self._epoch = epoch

        # Generate data.
        rng = random.Random(self.seed + self.epoch)
        instances_per_batch = self.global_batch_size // self.sequence_length
        total_instances = instances_per_batch * self.total_batches
        self._dataset = [
            torch.arange(start=start_idx, end=start_idx + self.sequence_length)
            for start_idx in (
                rng.randint(0, self.vocab_size - self.sequence_length - 2)
                for _ in range(total_instances)
            )
        ]

    def get_mock_batch(self) -> Dict[str, Any]:
        num_instances = self.rank_batch_size // self.sequence_length
        input_ids = torch.randint(0, self.vocab_size, (num_instances, self.sequence_length))
        return {"input_ids": input_ids}

    def _iter_batches(self) -> Iterable[Dict[str, Any]]:
        assert self._dataset is not None, "did you forget to call 'reshuffle()'?"

        # Get global batch instance indices. Shape: (total batches, instances per batch)
        instances_per_batch = self.global_batch_size // self.sequence_length
        indices = torch.arange(len(self._dataset)).view(self.total_batches, instances_per_batch)

        # Offset by batches processed so far.
        indices = indices[self.batches_processed :]

        for batch_indices in indices:
            # Slice batch indices up by rank to create data parallel micro-batches.
            local_batch_indices = batch_indices[self.dp_rank :: self.dp_world_size]
            yield self.collator([self._dataset[idx] for idx in local_batch_indices])

@dataclass
class CustomDataLoaderConfig(NumpyDataLoaderConfig):

    global_batch_size: int
    seed: int
    work_dir: Optional[str] = None
    num_threads: Optional[int] = None
    num_workers: int = 0
    prefetch_factor: Optional[int] = None
    target_device_type: Optional[str] = None
    sequence_length: Optional[int] = None
    vocab_size: Optional[int] = None

    def build(
        self,
        dataset,
        *,
        collator = None,
        mesh = None,
        dp_process_group = None,
        sequence_length = None,
        vocab_size = None,
    ):
        return CustomDataLoader(
            sequence_length=self.sequence_length,
            vocab_size=self.vocab_size,
            work_dir=self.work_dir,
            global_batch_size=self.global_batch_size,
            seed=self.seed,
        )
    

@dataclass
class ExperimentConfig(Config):
    model: TransformerConfig
    optim: AdamWConfig
    dataset: NumpyDatasetConfig
    data_loader: NumpyDataLoaderConfig
    trainer: TrainerConfig
    init_seed: int = 12536


def build_config(run_name: str, overrides: List[str]) -> ExperimentConfig:
    tokenizer_config = TokenizerConfig.gpt2()

    model_config = TransformerConfig.llama2_271M(
        vocab_size=tokenizer_config.padded_vocab_size(),  # a little bigger than actual vocab size to make it a multiple of 128
        compile=True,
        fused_ops=False,
        use_flash=False,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp, param_dtype=DType.bfloat16, reduce_dtype=DType.float32
        ),
    )

    optim_config = AdamWConfig(
        lr=1e-3,
        group_overrides=[
            OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
        ],
    )

    dummy_dataset_config = NumpyDatasetConfig.glob(
        "random1024.npy",  # can be globs
        name=NumpyDatasetType.fsl,
        sequence_length=1024,
        max_target_sequence_length=8192,
        tokenizer=tokenizer_config,
        work_dir=data_root+"/dataset-cache",
    )

    data_loader_config = CustomDataLoaderConfig(
        global_batch_size=256 * 1024,
        sequence_length=1024,
        vocab_size=tokenizer_config.padded_vocab_size(),
        seed=0,
        num_workers=4,
    )

    trainer_config = (
        TrainerConfig(
            save_folder=f"{data_root}/{run_name}",
            rank_microbatch_size=16 * 1024,
            save_overwrite=True,
            metrics_collect_interval=5,
            cancel_check_interval=5,
            load_key_mapping={
                # For backwards compatibility when loading older checkpoints.
                "lm_head.w_out.weight": "w_out.weight",
                "lm_head.norm.weight": "norm.weight",
            },
        )
        .with_callback("lr_scheduler", SchedulerCallback(scheduler=CosWithWarmup(warmup_steps=100)))
        .with_callback(
            "seq_len_scheduler",
            SequenceLengthSchedulerCallback(
                min_sequence_length=128, warmup_steps=100, enabled=False
            ),
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback("grad_clipper", GradClipperCallback(max_grad_norm=1.0))
        .with_callback(
            "checkpointer",
            checkpointer_callback
        )
        .with_callback(
            "comet",
            CometCallback(
                name=run_name,
                cancel_check_interval=10,
                enabled=False,  # change to true to enable
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=run_name,
                cancel_check_interval=10,
                enabled=True,  # change to true to enable
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback("profiler", ProfilerCallback(enabled=False))
    )

    return ExperimentConfig(
        model=model_config,
        optim=optim_config,
        dataset=dummy_dataset_config,
        data_loader=data_loader_config,
        trainer=trainer_config,
    ).merge(overrides)


def main(run_name: str, overrides: List[str]):
    config = build_config(run_name, overrides)

    # Set RNG states on all devices.
    seed_all(config.init_seed)

    device = get_default_device()

    # Build the world mesh, if needed.
    world_mesh = config.model.build_mesh(device=device)

    # Build components.
    model = config.model.build(
        init_device="meta",
        device=device,
        max_seq_len=config.dataset.sequence_length,
        mesh=world_mesh,
    )
    optim = config.optim.build(model)
    dataset = config.dataset.build()
    data_loader = config.data_loader.build(dataset, mesh=world_mesh)
    trainer = config.trainer.build(model, optim, data_loader, mesh=world_mesh)

    # Save config to W&B and each checkpoint dir.
    config_dict = config.as_config_dict()
    cast(CometCallback, trainer.callbacks["comet"]).config = config_dict
    cast(WandBCallback, trainer.callbacks["wandb"]).config = config_dict
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

    # Train.
    trainer.fit()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} run_name [OVERRIDES...]")
        sys.exit(1)

    run_name, *overrides = sys.argv[1:]

    prepare_training_environment()
    try:
        main(run_name, overrides=overrides)
    finally:
        teardown_training_environment()
