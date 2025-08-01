import torch
import torch.nn as nn
import torchmetrics
from torchmetrics.classification import MulticlassF1Score, MulticlassConfusionMatrix, MulticlassPrecision, MulticlassRecall, Accuracy
from torchmetrics.segmentation import GeneralizedDiceScore

import torch.distributed as dist
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix, f1_score 

from typing import Dict, Any

#from torchsummary import summary
import pytorch_lightning as pl

from finetune_util import InputMLP, InputConv
#from finetune_util import freeze_encoder_layers, LoRALinear, recursively_replace_lora

import deepspeed
from pytorch_caney.optimizers.build import build_optimizer
from pytorch_caney.transforms.abi_toa import AbiToaTransform
from pytorch_caney.models import ModelFactory
from typing import Tuple

# for visualizing
import matplotlib
matplotlib.use('Agg')  # headless backend
import matplotlib.pyplot as plt
import os


class DiceLoss(nn.Module):
    def __init__(self,smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self,logits,targets):
        num_classes = logits.shape[1]
        targets_ohe = F.one_hot(targets,num_classes).permute(0,3,1,2).float()
        probs = torch.softmax(logits,dim=1)
        dims = (0,2,3)
        intersection = torch.sum(probs * targets_ohe,dims)
        cardinality = torch.sum(probs + targets_ohe, dims)
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        return 1. - dice_score.mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # e.g., weights tensor
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=alpha, reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# test    LoRA Class    #
class LoRALinear(nn.Module):
    '''
    Pasted from https://docs.pytorch.org/torchtune/0.4/tutorials/lora_finetune.html
    This class can be called like LoRALinear(in_dim, out_dim, <module from satvision>, r, a, 0)
    to replace layers in satvision, can do module = LoRALinear(...)
    '''
    def __init__(self, in_dim: int, out_dim: int, orig_layer, rank: int, alpha: float, dropout: float):
        super().__init__()

        # This is the weights/linear layer from the original pretrained model
        self.orig = orig_layer

        # These are the new LoRA params. In general rank << in_dim, out_dim
        self.lora_a = nn.Linear(in_dim, rank, bias=False)
        self.lora_b = nn.Linear(rank, out_dim, bias=False)

        # Rank and alpha are commonly-tuned hyperparameters
        self.rank = rank
        self.alpha = alpha
        self.scaling = self.alpha / self.rank

        #self.weight = orig_layer.weight

        # Most implementations also include some dropout
        self.dropout = nn.Dropout(p=dropout)

        # The original params are frozen, and only LoRA params are trainable.
        self.orig.weight.requires_grad = False
        self.lora_a.weight.requires_grad = True
        self.lora_b.weight.requires_grad = True


    @property
    def weight(self) -> torch.Tensor: # added type hint
        # Combine base weight with the A/B weights
        delta_w = self.scaling * (self.lora_b.weight @ self.lora_a.weight)
        #print(f"[LoRALinear] ΔW mean abs: {delta_w.abs().mean().item():.6f}")
        #combined = self.orig.weight + delta_w
        return self.orig.weight + delta_w #

    def __getattr__(self, name):
        """Explicitly handle 'weight' attribute for PyTorch compatibility"""
        if name == 'weight':
            return self.weight # Resolve via property
        return super().__getattr__(name)





    @staticmethod
    def _new_recursively_replace_lora(model, target_module_paths=['attn.proj', 'attn.qkv'], rank=4, alpha=16, dropout=0.0):
        """
        Replace specific nn.Linear layers in the model by matching full module paths like 'attn.proj'
        """
        for name, module in model.named_modules():
            if any(name.endswith(target) for target in target_module_paths) and isinstance(module, nn.Linear):
                parent_name = ".".join(name.split(".")[:-1])
                attr_name = name.split(".")[-1]
                parent = model.get_submodule(parent_name)
                lora_layer = LoRALinear(module.in_features, module.out_features, module, rank, alpha, dropout)
                setattr(parent, attr_name, lora_layer)
                print(f"Replaced {name} with LoRA.")


        # ------
    # LoRA Helper ( original im testing new version - Danielle )
    # ------
    #@staticmethod
    def _recursively_replace_lora(module, target_layer_names=['qkv'], rank=4, alpha=16, dropout=0.0):
        """
        Recursively go through the model, replace layers named `target_layer_name`
        with LoRALinear wrapped layers.
        """
        for name, child in module.named_children():
            # If this child has the target layer name, replace it
            if name in target_layer_names and isinstance(child, nn.Linear):
                in_dim = child.in_features
                out_dim = child.out_features
                # Wrap the original layer with LoRA
                lora_layer = LoRALinear(in_dim, out_dim, child, rank=rank, alpha=alpha, dropout=dropout)
                setattr(module, name, lora_layer) # this replaces the module with the created lora layer
                print(f"Replaced layer '{name}' in \n{module}\n with LoRA")
            else:
                # Recursively check child modules
                LoRALinear._recursively_replace_lora(child, target_layer_names, rank, alpha, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # seems like Swin v2 code doesn't actually call this but we'll leave here for now
        # This would be the output of the original model
        frozen_out = self.orig(x)

        # lora_a projects inputs down to the much smaller self.rank,
        # then lora_b projects back up to the output dimension
        lora_out = self.lora_b(self.lora_a(self.dropout(x)))

        # Finally, scale by the alpha parameter (normalized by rank)
        # and add to the original model's outputs
        return frozen_out + (self.alpha / self.rank) * lora_out



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
        
        # ===== Set up preprocessor =====
        hidden_dims = self.config.MODEL.PREPROCESSOR.HIDDEN_DIMS

        if self.config.MODEL.PREPROCESSOR.NAME == 'mlp':
            self.preprocessor = InputMLP(in_channels=self.config.MODEL.IN_CHANS, out_channels=14, hidden_dims=hidden_dims)
        elif self.config.MODEL.PREPROCESSOR.NAME == 'conv':
            self.preprocessor = InputConv(in_channels=16, out_channels=14, hidden_dims=hidden_dims,kernel_size=self.config.MODEL.PREPROCESSOR.CONV_KERNEL_SIZE)
        else:
            print("config.MODEL.PREPROCESSOR.NAME must be mlp or conv, using Identity as preprocessor")
            self.preprocessor = nn.Identity()

        # ===== Set up encoder ======
        # VPT will be enabled when encoder is initialized below, if self.config.VPT == True
        self.encoder = factory.get_component(component_type="encoder",
            name=self.config.MODEL.ENCODER,
            config=self.config)

        # Freeze up to MODEL.FREEZE_LAYER
        self._freeze(self.config.MODEL.FREEZE_LAYER)

        # Freeze all parameters (except VPT prompts) 
        if self.config.MODEL.FREEZE_ALL:
            for param in self.encoder.parameters():
                param.requires_grad = False
        # Unfreeze the prompt embeddings
        if self.config.MODEL.VPT:
            self.encoder.model.prompt_embeddings.requires_grad = True

        # Enable LoRA
        if self.config.MODEL.LORA:
            # Config only sets if lora is used or not and rank, must edit target_layer_names, alpha, etc for now
            LoRALinear._new_recursively_replace_lora(self.encoder, 
                    target_module_paths=['qkv','attn.proj'], 
                    rank=self.config.MODEL.LORA_RANK, 
                    alpha=2*self.config.MODEL.LORA_RANK, 
                    dropout=0.1)
        print(f"LoRA dropout = 0.1") 

        ### Check freezing
        #print("================== encoder.named_parameters() =====================")
        #for name, param in self.encoder.named_parameters():
        #    print(name, param.requires_grad)


        # ===== Set up decoder and segmentation head =====
        self.decoder = factory.get_component(
            component_type="decoder",
            name=self.config.MODEL.DECODER,
            num_features=self.encoder.num_features)

        self.segmentation_head = factory.get_component(
            component_type="head",
            name="segmentation_head",
            decoder_channels=self.decoder.output_channels,
            num_classes=self.NUM_CLASSES,
            output_shape=self.OUTPUT_SHAPE)

        # ===== Put the model all together =====
        self.model = nn.Sequential(self.preprocessor,
                self.encoder,
                self.decoder,
                self.segmentation_head)

    def configure_losses(self):
        loss: str = self.config.LOSS.NAME

        counts = torch.tensor([41450105, 108357310, 26058977, 8775999, 60675241], dtype=torch.float32)

        # Inverse
        weights = 1.0 / counts
        weights = weights / weights.sum() * len(counts)  # Normalize: mean weight ≈ 1.0

        if loss == "CEDice":
            self.dice_loss = DiceLoss()
            self.dice_weight = .26
            self.ce_loss = nn.CrossEntropyLoss(weight=weights)
        elif loss == 'ce':
            if True: # set to false to disable weighted ce loss
                self.criterion = nn.CrossEntropyLoss(weight=weights)
            else:
                self.criterion = nn.CrossEntropyLoss()
        elif loss == 'Dice' or loss == "dice":
            self.criterion = GeneralizedDiceScore(num_classes=5)
        elif loss == "focal" or loss == "Focal":
            self.criterion = FocalLoss(alpha=weights,gamma=2)
        else:
            raise ValueError(
                f'Loss type "{loss}" is not valid. '
                'Currently supports "ce", "dice", or "focal".'
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
        
        self.test_f1 = MulticlassF1Score(num_classes=num_classes, average='none')
        self.test_confmat = MulticlassConfusionMatrix(num_classes=num_classes)
        self.test_precision = MulticlassPrecision(num_classes=num_classes, average='none')
        self.test_recall = MulticlassRecall(num_classes=num_classes, average='none')
        self.test_accuracy = Accuracy(task="multiclass",num_classes=num_classes)


    def forward(self, x):
        return self.model(x)


    def training_step(self, batch, batch_idx):
 
        inputs, targets = batch

        logits = self.forward(inputs)

        # resize logits output
        logits = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)


        preds = torch.argmax(logits, dim=1)
        loss = self.config.LOSS.NAME

        if loss == "ce" or loss == "focal" or loss == "Focal":
            train_loss = self.criterion(logits, targets.long())
        elif loss == "dice" or loss== "Dice":
            probs = torch.softmax(logits, dim=1)
            target_ohe = F.one_hot(targets.long(), num_classes=probs.shape[1])  # [B, H, W, C]
            target_ohe = target_ohe.permute(0, 3, 1, 2).float()  # [B, C, H, W]
            train_loss = self.criterion(probs, target_ohe)
        elif loss == "CEDice":  
            ce = self.ce_loss(logits,targets.long())
            dice = self.dice_loss(logits,targets.long())
            train_loss = self.dice_weight * dice + (1 - self.dice_weight) * ce


        iou = self.train_iou(preds, targets)

        iou_per_class = self.train_iou_per_class(preds, targets)
        for i, class_iou in enumerate(iou_per_class):
            self.log(f'train_iou_{i}', class_iou, on_step=False, on_epoch=True)

        self.train_loss_avg.update(train_loss)
        self.train_iou_avg.update(iou)
        self.log('train_loss', self.train_loss_avg.compute(),on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_iou', self.train_iou_avg.compute(),on_step=False, on_epoch=True, prog_bar=True)

        return train_loss

    def validation_step(self, batch, batch_idx):

        inputs, targets = batch
        
        #print(f"[validation_step] input shape: {inputs.shape}")

        logits = self.forward(inputs)

        logits = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)
        
        preds = torch.argmax(logits, dim=1)
        
        loss = self.config.LOSS.NAME
        if loss == "ce" or loss == "focal" or loss == "Focal":
            val_loss = self.criterion(logits, targets.long())
        elif loss == "dice" or loss == "Dice":
            probs = torch.softmax(logits, dim=1)
            target_ohe = F.one_hot(targets.long(), num_classes=probs.shape[1])  # [B, H, W, C]
            target_ohe = target_ohe.permute(0, 3, 1, 2).float()  # [B, C, H, W]
            val_loss = self.criterion(probs, target_ohe)
        elif loss == "CEDice":
            ce = self.ce_loss(logits,targets.long())
            dice = self.dice_loss(logits,targets.long())
            val_loss = self.dice_weight * dice + (1 - self.dice_weight) * ce

        


        val_iou = self.val_iou(preds, targets)


        iou_per_class = self.val_iou_per_class(preds, targets)
        for i, class_iou in enumerate(iou_per_class):
            self.log(f'val_iou_{i}', class_iou, on_step=False, on_epoch=True)

        self.val_loss_avg.update(val_loss)
        self.val_iou_avg.update(val_iou)

        self.log('val_loss', self.val_loss_avg.compute(),
                    on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_iou', self.val_iou_avg.compute(),
                    on_step=False, on_epoch=True, prog_bar=True, sync_dist=True) 

        return val_loss

    def test_step(self, batch, batch_idx):

        inputs, targets = batch
        logits = self.forward(inputs)
        
        # Reshape logits
        logits = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)

        preds = torch.argmax(logits,dim=1)
        
        loss = self.config.LOSS.NAME
        if loss == "ce" or loss == "focal" or loss == "Focal":
            test_loss = self.criterion(logits, targets.long())
        elif loss == "dice" or loss == "Dice":
            probs = torch.softmax(logits, dim=1)
            target_ohe = F.one_hot(targets.long(), num_classes=probs.shape[1])  # [B, H, W, C]
            target_ohe = target_ohe.permute(0, 3, 1, 2).float()  # [B, C, H, W]
            test_loss = self.criterion(probs, target_ohe)
        elif loss == "CEDice":
            ce = self.ce_loss(logits,targets.long())
            dice = self.dice_loss(logits,targets.long())
            test_loss = self.dice_weight * dice + (1 - self.dice_weight) * ce


        test_iou = self.test_iou(preds, targets)

        self.log(f'test_iou', test_iou, on_step=False, on_epoch=True)
        
        # val_iou_per_class is just an iou metric that's defined with that name. could define one for testing but it doesn't matter.
        iou_per_class = self.val_iou_per_class(preds, targets)
        for i, class_iou in enumerate(iou_per_class):
            self.log(f'test_iou_{i}', class_iou, on_step=False, on_epoch=True)

        self.test_f1.update(preds.flatten(), targets.flatten())
        self.test_confmat.update(preds.flatten(), targets.flatten())
        self.test_precision.update(preds.flatten(), targets.flatten())
        self.test_recall.update(preds.flatten(), targets.flatten())
        self.test_accuracy.update(preds.flatten(),targets.flatten())

        # ===== Visualize =====
    
        if batch_idx == 1 and self.global_rank == 0:  # only save for the first batch on rank 0
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

                im0 = axs[0].imshow(pred_mask, cmap=cmap_phase,vmin=-0.5,vmax=4.5)
                axs[0].set_title("Prediction")
                axs[0].axis("off")

                im1 = axs[1].imshow(truth_mask, cmap=cmap_phase, vmin=-0.5, vmax=4.5)
                axs[1].set_title("Ground Truth")
                axs[1].axis("off")

                cbar = fig.colorbar(im1, ax=axs.ravel().tolist(), orientation='horizontal',
                        ticks=tick_vals, fraction=0.1, pad=0.1)
                cbar.ax.set_xticklabels(tick_labels, fontsize=8)

                fig.suptitle(f"Test Predictions Sample {i}", fontsize=12)
                plt.tight_layout()

                # get job id, or if running locally outputs "unknown job"
                job_id = os.environ.get("SLURM_JOB_ID", "unknown_job")
                save_dir = os.path.join("CloudPhaseVisuals", f"job_{job_id}",self.config.TAG)
                os.makedirs(save_dir, exist_ok=True)


                plt.savefig(os.path.join(save_dir, f"example{i}.png"))
                plt.close()

                # --- Get and log SwinV2 embedding statistics ---
                # Call the method on swinV2 encoder instance
                all_stage_stats = self.encoder.model.get_all_stage_block_statistics()

                # Log the statistics
                for stage_name, stage_data in all_stage_stats.items():
                    if stage_data:
                        for block_name, block_stats in stage_data.items():
                            # Log mean and std dev for each captured block
                            self.log(f'stage {stage_name}/{block_name}_mean', block_stats['mean'], on_step=False, on_epoch=True)
                            self.log(f'stage {stage_name}/{block_name}_std', block_stats['std'], on_step=False, on_epoch=True)
                    else:
                        # This case should ideally not happen if BasicLayer is capturing outputs
                        print(f"Warning: No block outputs captured for {stage_name} in test_step.")

        return test_loss


    def on_test_epoch_end(self):
        f1_per_class = self.test_f1.compute()
        for i, f1 in enumerate(f1_per_class):
            self.log(f'test_f1_score_{i}', f1)

        precision_per_class = self.test_precision.compute()
        recall_per_class = self.test_recall.compute()
        
        for i in range(self.NUM_CLASSES):
            self.log(f'test_precision_{i}', precision_per_class[i])
            self.log(f'test_recall_{i}', recall_per_class[i])


        self.log("test_acc", self.test_accuracy.compute(), prog_bar=True)

        confmat = self.test_confmat.compute()
        print("Confusion Matrix:\n", confmat.cpu().numpy())

        # Reset metrics for safety
        self.test_f1.reset()
        self.test_confmat.reset()


    def configure_optimizers(self):
        optimizer = build_optimizer(self.config, self.model, is_pretrain=False)
        print(f'Using optimizer: {optimizer}')
        return optimizer

    def on_train_epoch_start(self):
        self.train_loss_avg.reset()
        self.train_iou_avg.reset()

        current_epoch = self.current_epoch
        
        if current_epoch == 20:
            for name, param in self.encoder.model.named_parameters():
                if "layers.3" in name:
                    print(f"Unfreezing {name} at epoch 20")
                    param.requires_grad = True

    def on_validation_epoch_start(self):
        self.val_loss_avg.reset()
        self.val_iou_avg.reset()
                                                       

