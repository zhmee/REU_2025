import argparse 
import os
import glob
import torch

import torch.distributed as dist
from torch.utils.data import DataLoader

from lightning.pytorch import Trainer
from pytorch_lightning.loggers import CSVLogger
from lightning.pytorch.strategies import FSDPStrategy
from lightning.pytorch.callbacks import ModelCheckpoint


from pytorch_caney.configs.config import _C, _update_config_from_file
from pytorch_caney.utils import get_strategy, get_distributed_train_batches
from pytorch_caney.pipelines import PIPELINES, get_available_pipelines
from pytorch_caney.datamodules import DATAMODULES, get_available_datamodules
from pytorch_caney.transforms.abi_toa import AbiToaTransform



from finetune_util import CloudPhaseDataset
import os
import subprocess

#cwd = os.getcwd()
#print(f"Current working directory: {cwd}")
#subprocess.run(['df', '-h', cwd])

# Also check /tmp space
#subprocess.run(['df', '-h', '/tmp'])

def main(config, output_dir):
    # Get the proper pipeline
    available_pipelines = get_available_pipelines()
    #print("Available pipelines:", available_pipelines)
    pipeline = PIPELINES[config.PIPELINE]

    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        print(f'Using {pipeline}')
    
    ptlPipeline = pipeline(config)
    
    # Resume from checkpoint
    if config.MODEL.RESUME:
        print_cuda_memory("before load from checkpoint")
        print(f'Attempting to resume from checkpoint {config.MODEL.RESUME}')
        ptlPipeline = pipeline.load_from_checkpoint(config.MODEL.RESUME)

    logger = CSVLogger(save_dir="PhasePredLogs",name=f"phasepred_{config.TAG}")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_iou",
        mode="max",
        save_top_k=1,
        filename="best-miou-checkpoint",
        verbose=True,
        save_weights_only=True,
        dirpath="/umbc/rs/cybertrn/team1/research/Danielle/pytorch-caney/PhasePredLogs/checkpoints"
    )
    
    strategy = FSDPStrategy(state_dict_type="sharded")

    trainer = Trainer(
        accelerator=config.TRAIN.ACCELERATOR,
        strategy=strategy,
        precision=config.PRECISION,
        logger=logger,
        max_epochs=config.TRAIN.EPOCHS,
        log_every_n_steps=config.PRINT_FREQ,
        devices=4,
        num_nodes=2,
        default_root_dir=output_dir,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback]
    )

    if config.TRAIN.LIMIT_TRAIN_BATCHES:
        trainer.limit_train_batches = get_distributed_train_batches(
            config, trainer)

    npz_paths = sorted(glob.glob("/umbc/rs/nasa-access/users/xingyan/pytorch-caney/data/downstream_2dcloud/*.npz"))
   
    in_chans = config.MODEL.IN_CHANS # (14 or 16)

    transform = AbiToaTransform(img_size=128)

    full_ds = CloudPhaseDataset(npz_paths,in_chans=in_chans,transform=transform)
    train_ds, val_ds, test_ds  = torch.utils.data.random_split(full_ds, [0.70,0.20,0.10])

    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=config.DATA.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.DATA.BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=config.DATA.BATCH_SIZE)

    # Train
    trainer.fit(model=ptlPipeline,train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    
    # === Reload the best model from checkpoint ===
    best_ckpt_path = checkpoint_callback.best_model_path
    print(f"Best checkpoint path: {best_ckpt_path}")


    # === Evaluate on test set ===
    test_results = trainer.test(model, dataloaders=test_loader,ckpt_path=best_ckpt_path)
    print("TEST RESULTS:", test_results)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config-path', type=str, help='Path to pretrained model config')

    hparams = parser.parse_args()

    config = _C.clone()
    _update_config_from_file(config, hparams.config_path)

    output_dir = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)

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
    
