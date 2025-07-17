import torch
import torch.nn as nn
import torchmetrics
import torch.distributed as dist
import torch.nn.functional as F

#from torchsummary import summary

import lightning.pytorch as pl

from pytorch_caney.optimizers.build import build_optimizer
from pytorch_caney.transforms.abi_toa import AbiToaTransform
from pytorch_caney.models import ModelFactory
from typing import Tuple

# for visualizing
import matplotlib
matplotlib.use('Agg')  # headless backend
import matplotlib.pyplot as plt
import os



class InputMLP(nn.Module):
    def __init__(self, in_channels=16, out_channels=14, hidden_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_channels,hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_channels))

    def forward(self,x):
        # x is (B, 16, 128, 128)
        #print(f"[ChannelMLP] x initial shape: {x.shape}")
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)         # (B, 128, 128, 16)
        
        #print(f"x shape before mlp: {x.shape}")

        x = self.mlp(x)
        x = x.permute(0, 3, 1, 2)         # (B, 14, 128, 128) – convert back to format the encoder expects
        return x





class PhasePred(pl.LightningModule):
    NUM_CLASSES: int = 5
    OUTPUT_SHAPE: Tuple[int, int] = (91, 40)

    def _freeze(self, depth):
        '''
        depth : -1 -> freezes nothing
        depth : 0 -> freezes only patch_embed
        depth : n -> patch_embed + layer[0:n+1]
        '''

        #check patch embed first
        for param in self.encoder.model.patch_embed.parameters():
            param.requires_grad = (depth < 0)

        for index, stage in enumerate(self.encoder.model.layers):
            freeze = (depth >= 0) and (index <= depth) # 
            for param in stage.parameters():
                param.requires_grad = not freeze # (if yes freeze, no require grad)


    def __init__(self,config):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.config = config
        self.configure_models()
        self.configure_losses()
        self.configure_metrics()
        self.transform = AbiToaTransform(self.config)

    def configure_models(self):
        factory = ModelFactory()
        
        hidden_dim = 32

        print(f"Using hidden dim size {hidden_dim} for MLP")
        self.input_mlp = InputMLP(in_channels=self.config.MODEL.IN_CHANS, out_channels=14,hidden_dim=hidden_dim)


        #for name, param in self.preprocessor.named_parameters():
        #    print(name, param.shape, param.requires_grad)


        self.encoder = factory.get_component(component_type="encoder",
            name=self.config.MODEL.ENCODER,
            config=self.config)

        
        # Pick freeze depth in configs/default.yaml
        # TODO stop hardcoding this
        #self._freeze(3)
        #print("froze up to layer 3")

        self.decoder = factory.get_component(
            component_type="decoder",
            name=self.config.MODEL.DECODER,
            num_features=self.encoder.num_features)

        #if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        #    print("encoder summary")
        #    print(self.encoder)

        self.segmentation_head = factory.get_component(
            component_type="head",
            name="segmentation_head",
            decoder_channels=self.decoder.output_channels,
            num_classes=self.NUM_CLASSES,
            output_shape=self.OUTPUT_SHAPE)

        
        #print(self.decoder)
        #print(self.segmentation_head)
        self.model = nn.Sequential(self.input_mlp,
                self.encoder,
                self.decoder,
                self.segmentation_head)
        #print(self.model)
        #if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        #    print("Printing all model parameters")
        #    for name, param in self.named_parameters():
        #        print(name, param.shape, param.requires_grad)

    def configure_losses(self):
        loss: str = self.config.LOSS.NAME
        if loss == 'ce':

            if True: # set to false to disable weighted ce loss
                counts = torch.tensor([23867479, 67026240, 26871247, 7557767, 21510675], dtype=torch.float32)
                

                # Inverse
                weights = 1.0 / counts
                weights[3] = weights[3] 
                weights = weights / weights.sum() * len(counts)  # Normalize: mean weight ≈ 1.0

                self.criterion = nn.CrossEntropyLoss(weight=weights)
            else:
                self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(
                f'Loss type "{loss}" is not valid. '
                'Currecntly supports "ce".'
            )
    
    def configure_metrics(self):
        # Very similar to configure_metrics from 3dcloud task, changed task from binary to multiclass

        num_classes = self.NUM_CLASSES
        

        self.test_iou = torchmetrics.JaccardIndex(num_classes=num_classes,task="multiclass")
        self.test_iou_per_class = torchmetrics.JaccardIndex(num_classes=num_classes, task="multiclass", average=None)

        self.train_iou = torchmetrics.JaccardIndex(num_classes=num_classes,task="multiclass")
        self.train_iou_per_class = torchmetrics.JaccardIndex(num_classes=num_classes, task="multiclass", average=None)

        self.val_iou = torchmetrics.JaccardIndex(num_classes=num_classes,task="multiclass")
        self.val_iou_per_class = torchmetrics.JaccardIndex(num_classes=num_classes, task="multiclass", average=None)

        self.train_loss_avg = torchmetrics.MeanMetric()
        self.val_loss_avg = torchmetrics.MeanMetric()

        self.train_iou_avg = torchmetrics.MeanMetric()
        self.val_iou_avg = torchmetrics.MeanMetric()

    def forward(self, x):
        #x = self.preprocessor(x)
        #x = self.encoder(x)
        #x = self.decoder(x)
        #x = self.segmentation_head(x)
        return self.model(x)


    def training_step(self, batch, batch_idx):

        inputs, targets = batch

        logits = self.forward(inputs)  
        
        # resize logits output
        logits = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)

        loss = self.criterion(logits, targets.long()) 
          
        preds = torch.argmax(logits, dim=1)
        iou = self.train_iou(preds, targets)

        iou_per_class = self.train_iou_per_class(preds, targets)
        for i, class_iou in enumerate(iou_per_class):
            self.log(f'train_iou_{i}', class_iou, on_step=False, on_epoch=True)

        self.train_loss_avg.update(loss)
        self.train_iou_avg.update(iou)
        self.log('train_loss', self.train_loss_avg.compute(),on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_iou', self.train_iou_avg.compute(),on_step=False, on_epoch=True, prog_bar=True)
        # I set on step to false just for less noise to make sure things work

        return loss

    def validation_step(self, batch, batch_idx):

        inputs, targets = batch
        
        print(f"[validation_step] input shape: {inputs.shape}")

        logits = self.forward(inputs)

        logits = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)

        val_loss = self.criterion(logits, targets.long())
        preds = torch.argmax(logits, dim=1)
            
        val_iou = self.val_iou(preds, targets.int())

        iou_per_class = self.val_iou_per_class(preds, targets)
        for i, class_iou in enumerate(iou_per_class):
            self.log(f'val_iou_{i}', class_iou, on_step=False, on_epoch=True)

        self.val_loss_avg.update(val_loss)
        self.val_iou_avg.update(val_iou)

        self.log('val_loss', self.val_loss_avg.compute(),
                    on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_iou', self.val_iou_avg.compute(),
                    on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # Visualize
        if self.current_epoch > 3 and  batch_idx < 2 and self.global_rank == 0:  # only save for the first few batches on rank 0
            os.makedirs("PredVisuals", exist_ok=True)
            phase_spec = [
                (0, '#3498db', 'Clear_sky'),      # Blue
                (1, '#2ecc71', 'Liquid_water'),   # Green
                (2, '#27ae60', 'Supercooled'),    # Dark Green
                (3, '#f39c12', 'Mixed'),          # Orange
                (4, '#e74c3c', 'Ice')            # Red
            ]
            cmap_phase = plt.cm.colors.ListedColormap([c[1] for c in phase_spec])
            tick_vals = [c[0] for c in phase_spec]
            tick_labels = [c[2] for c in phase_spec]
            
            for i in range(min(2, inputs.size(0))):  # visualize 2 examples max
                pred_mask = preds[i].cpu()
                truth_mask = targets[i].cpu()
                
                fig, axs = plt.subplots(1,2, figsize=(10,4))

                im0 = axs[0].imshow(pred_mask, cmap=cmap_phase,vmin=-0.5,vmax=5.5)
                axs[0].set_title("Prediction")
                axs[0].axis("off")

                im1 = axs[1].imshow(truth_mask, cmap=cmap_phase, vmin=-0.5, vmax=5.5)
                axs[1].set_title("Ground Truth")
                axs[1].axis("off")

                cbar = fig.colorbar(im1, ax=axs.ravel().tolist(), ticks=tick_vals, shrink=0.8,fraction=0.046, pad=0.04)
                cbar.ax.set_yticklabels(tick_labels, fontsize=8)

                fig.suptitle(f"Epoch {self.current_epoch} | Batch {batch_idx} | Sample {i}", fontsize=12)
                plt.tight_layout()
                
                # get job id, or if running locally outputs "unknown job"
                job_id = os.environ.get("SLURM_JOB_ID", "unknown_job")
                save_dir = os.path.join("PredVisuals", f"job_{job_id}")
                os.makedirs(save_dir, exist_ok=True)
                

                plt.savefig(os.path.join(save_dir, f"epoch{self.current_epoch}_batch{batch_idx}_example{i}.png"))
                plt.close()

                
        return val_loss

    def test_step(self, batch, batch_idx):
        
        #if batch_idx == 0:
        #    print("=== test_step model structure ===")
        #    print(self)
        #    print("=== Named parameters ===")
        #    for name, param in self.named_parameters():
        #        print(name, param.shape, param.requires_grad)

        inputs, targets = batch
        #print(f"[test_step] Input batch shape: {inputs.shape}")
        logits = self.forward(inputs)

        logits = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)

        test_loss = self.criterion(logits, targets.long())
        preds = torch.argmax(logits,dim=1)

        test_iou = self.test_iou(preds, targets.int())

        iou_per_class = self.test_iou_per_class(preds,targets)
        for i, class_iou in enumerate(iou_per_class):
            self.log(f'test_iou_{i}', class_iou, on_step=False, on_epoch=True)

        return test_loss

    def configure_optimizers(self):
        optimizer = build_optimizer(self.config, self.model, is_pretrain=True)
        print(f'Using optimizer: {optimizer}')
        return optimizer

    def on_train_epoch_start(self):
        self.train_loss_avg.reset()
        self.train_iou_avg.reset()

    def on_validation_epoch_start(self):
        self.val_loss_avg.reset()
                                                       


