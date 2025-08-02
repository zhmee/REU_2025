from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset
import sys, os
from pathlib import Path 

from pytorch_lightning.strategies import DeepSpeedStrategy

from pytorch_caney.models.encoders.swinv2 import SwinTransformerV2
from finetune_util import CloudDataset, TimerLogger


from collections import Counter


import torch
from torch.utils.data import WeightedRandomSampler
from pytorch_lightning import Trainer

from torch.utils.data import Dataset, DataLoader

from pytorch_caney.configs.config import _C, _update_config_from_file
from pytorch_caney.pipelines import PIPELINES
from phase_pred_pipeline import PhasePred
import pytorch_lightning as pl
assert issubclass(PhasePred, pl.LightningModule), "PhasePred is not a LightningModule"


import argparse
import glob
import torch.distributed as dist

from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, Timer
from pytorch_lightning.strategies import FSDPStrategy
from pytorch_caney.utils import get_distributed_train_batches


def main(config, output_dir):
    
    pipeline = PIPELINES[config.PIPELINE]
    print(f'Using {pipeline}')
    ptlPipeline = pipeline(config)

    # === Build LightningModule ===
    if config.MODEL.RESUME:
        ptlPipeline  = ptlPipeline.load_from_checkpoint(config.MODEL.RESUME)
   
    job_id = os.environ.get("SLURM_JOB_ID", "no_job_id")  # fallback for local runs
    log_path = os.path.join("Logs", config.PIPELINE, config.TAG, job_id)
    os.makedirs(log_path, exist_ok=True)        

    logger = CSVLogger(
            save_dir=log_path,
            name=""  # For instance, logs will go to Logs/PhasePred/config_tag/11111
            ) 
    if config.PIPELINE == "multitask2":
        monitor = "val_phase_iou"
        mode = "max"
    elif config.PIPELINE == "PhasePred" or config.PIPELINE == "CloudMask":
        monitor = "val_iou"
        mode = "max"
    else:
        monitor = "val_loss"
        mode = "min"

    
    checkpoint_callback = ModelCheckpoint(
        monitor=None,#monitor, #mode=mode,
        save_top_k=1,
        filename="best-miou-checkpoint",
        verbose=True,
        dirpath=log_path)

    timer_callback = Timer()
    timer_logger_callback = TimerLogger(timer_callback)

    callbacks = [checkpoint_callback, timer_callback, timer_logger_callback]

    #strategy = FSDPStrategy(state_dict_type="sharded")
    strategy = DeepSpeedStrategy(stage=3,config="deepspeed_config.json")

    trainer = Trainer(
        accelerator=config.TRAIN.ACCELERATOR,
        strategy=strategy,
        precision=config.PRECISION,
        logger=logger,
        max_epochs=config.TRAIN.EPOCHS,
        devices=2,
        num_nodes=1,
        log_every_n_steps=config.PRINT_FREQ,
        fast_dev_run=False,
        default_root_dir=output_dir,
        enable_checkpointing=True,
        callbacks=callbacks
    )

    if config.TRAIN.LIMIT_TRAIN_BATCHES:
        trainer.limit_train_batches = get_distributed_train_batches(
            config, trainer)

    # NEW : TESTING ENVIRONMENTAL VARIABLE FOR DOWNSTREAM CLOUD DATA PATH  #
    CLOUD_ROOT = Path(os.environ.get("DS_2D_CLOUD_ROOT",""))
    if not CLOUD_ROOT.is_dir():
        raise RuntimeError(
                "ERROR: Downstream 2dCloud data root is not set\n"
                "Please configure in env_local.sh."
                )
    npz_paths = sorted(CLOUD_ROOT.glob("*.npz"))
    in_chans = config.MODEL.IN_CHANS # (14 or 16)
    # END NEW  #

    in_chans = config.MODEL.IN_CHANS # (14 or 16)

    # Setup datamodule
    full_ds = CloudDataset(npz_paths,in_chans = in_chans, task=config.PIPELINE)
    train_ds, val_ds, test_ds  = torch.utils.data.random_split(full_ds, [0.80,0.10,0.10])


    generator = torch.Generator().manual_seed(1)
    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=config.DATA.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.DATA.BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=config.DATA.BATCH_SIZE)


    # Train
    trainer.fit(model=ptlPipeline,train_dataloaders=train_loader, val_dataloaders=val_loader)    

    # === Reload the best model from checkpoint ===
    ckpt_dir = checkpoint_callback.best_model_path
    print(f"Best checkpoint path: {ckpt_dir}")
    
    ## I think you need to make a new instance of the ptlPipeline
    ptlPipeline = pipeline(config)

    strategy = DeepSpeedStrategy(stage=3,config="deepspeed_config.json",load_full_weights=False)
    test_trainer = Trainer(
        accelerator=config.TRAIN.ACCELERATOR,
        strategy=strategy,
        precision=config.PRECISION,
        devices=2,
        num_nodes=1,
        default_root_dir=output_dir
        )


    from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
    fp32_model = load_state_dict_from_zero_checkpoint(model=ptlPipeline,checkpoint_dir=ckpt_dir)
    test_trainer.test(model=fp32_model,dataloaders=test_loader)

    




if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config-path', type=str, help='Path to pretrained model config')

    hparams = parser.parse_args()

    config = _C.clone()
    _update_config_from_file(config, hparams.config_path)

    output_dir=os.path.join("Logs", config.PIPELINE, config.TAG)

    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        print(f'Output directory: {output_dir}')

    os.makedirs(output_dir, exist_ok=True)

    path = os.path.join(output_dir,
                        f"{config.TAG}.config.json")

    with open(path, "w") as f:
        f.write(config.dump())

    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        print(f"Full config saved to {path}")
        print(config.dump())

    main(config, output_dir)

