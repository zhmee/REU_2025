import torch
import torch.nn as nn
import torchmetrics
import torch.distributed as dist
import torch.nn.functional as F

import numpy as np
from torchmetrics import R2Score
from torchmetrics.classification import MulticlassF1Score, MulticlassConfusionMatrix, MulticlassPrecision, MulticlassRecall

from typing import Dict, Any
from sklearn.metrics import classification_report

#from torchsummary import summary
import pytorch_lightning as pl

from finetune_util import InputMLP, InputConv

from pytorch_caney.optimizers.build import build_optimizer
from pytorch_caney.transforms.abi_toa import AbiToaTransform
from pytorch_caney.models import ModelFactory
from pytorch_caney.models.decoders import SwinUNet
from typing import Tuple

# for visualizing
import matplotlib
matplotlib.use('Agg')  # headless backend
import matplotlib.pyplot as plt
import os


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3,1, 2).float()
        probs = torch.softmax(logits, dim=1)
        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_one_hot, dims)
        cardinality = torch.sum(probs + targets_one_hot, dims)
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



# LoRA Class    #
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

    # ------
    # LoRA Helper
    # ------
    @staticmethod
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
                #print(f"Replaced layer '{name}' in \n{module}\n with LoRA")
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

class CloudMultiTask2(pl.LightningModule):
    NUM_CLASSES: int = 2
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

        self.num_phase_classes = 5
        
        self.all_preds_phase = []
        self.all_targets_phase = []
        self.all_preds_mask = []
        self.all_targets_mask = []


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
        self.encoder = factory.get_component(component_type="encoder",
            name=self.config.MODEL.ENCODER,
            config=self.config)

        # Freeze up to MODEL.FREEZE_LAYER
        if self.config.MODEL.FREEZE_ALL:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self._freeze(self.config.MODEL.FREEZE_LAYER)

        if self.config.MODEL.LORA:
            # Config only sets if lora is used or not, must edit target_layer_names, rank, alpha, etc for now
            LoRALinear._recursively_replace_lora(self.encoder, target_layer_names='qkv', rank=self.config.MODEL.LORA_RANK, alpha=16, dropout=0.0)
        

        #TODO Can set these in config
        # mask phase cod cps
        self.m_out_chans = 64
        self.p_out_chans = 64
        self.cod_out_chans = 14
        self.cps_out_chans = 14


        ## Setup unet decoders
        self.unet = SwinUNet(self.encoder.model)
        
        # Setup heads
        self.mask_head = nn.Conv2d(64, 2, kernel_size=1)
        self.phase_head = nn.Conv2d(64 + 2, 5, kernel_size=1)  # +1 to include mask logits as input
        self.cod_head = nn.Conv2d(64 + 2, 1, kernel_size=1)
        self.cps_head = nn.Conv2d(64 + 2, 1, kernel_size=1)



    def configure_losses(self):
        loss: str = self.config.LOSS.NAME # don't use this rn lol
        
        # Define weights for cloud mask and cloud phase based on class counts
        maskcounts = torch.tensor([55085190,190232442], dtype=torch.float32)
        maskweights = 1.0 / (maskcounts)
        maskweights = maskweights / maskweights.sum() * len(maskcounts)  # Normalize: mean weight ≈ 1.0

        phasecounts = torch.tensor([41450105, 108357310, 26058977, 8775999, 60675241], dtype=torch.float32)
        phaseweights = 1.0 / phasecounts
        phaseweights = phaseweights / phaseweights.sum() * len(phasecounts)  # Normalize: mean weight ≈ 1.0

        self.mask_loss = FocalLoss(alpha=maskweights,gamma=2)

        self.dice_loss = DiceLoss()
        self.dice_weight = .25
        self.ce_loss = nn.CrossEntropyLoss(weight=weights)



        self.cod_loss = nn.MSELoss()
        self.cps_loss = nn.MSELoss()

        num_tasks = 4
        self.log_vars = nn.Parameter(torch.zeros(num_tasks)) # for uncertainty weighting

        self.losses = [self.mask_loss,
                self.phase_loss,
                self.cod_loss,
                self.cps_loss]

        # Was using this to  set up the loss weights, commented out because we don't need that for uncertainty weighting!
        self.mask_weight = torch.nn.Parameter(torch.tensor(1.0))  # Initialize to 10
        self.phase_weight = torch.nn.Parameter(torch.tensor(1.0))  # Initialize to 5
        self.cps_weight = torch.nn.Parameter(torch.tensor(.01))  # Initialize to 1
        self.cod_weight = torch.nn.Parameter(torch.tensor(.01))  # Initialize to 1
        ## Comment out this code and uncomment above to make loss weights trainable
        #self.mask_weight = torch.tensor(1.0)
        #self.phase_weight = torch.tensor(1.0)
        #self.cps_weight = torch.tensor(0.01)
        #self.cod_weight = torch.tensor(0.01)

    # This will be called to get loss during training, if using uncertainty weighting.
    def _calculate_total_uncertainty_loss(self, task_losses: list[torch.Tensor]):
        """
        Calculates the total loss using Kendall's Uncertainty Weighting.

        """

        total_loss = 0
        for i, task_loss in enumerate(task_losses):
            # Ensure task_loss is a scalar tensor
            if not isinstance(task_loss, torch.Tensor) or task_loss.dim() != 0:
                raise TypeError(f"Task loss at index {i} must be a scalar tensor, but got {task_loss.shape}")

            sigma_squared = torch.exp(self.log_vars[i])
            term = (task_loss / (2 * sigma_squared)) + (0.5 * self.log_vars[i])
            total_loss += term
        return total_loss



    def configure_metrics(self):
        num_classes = self.NUM_CLASSES
        
        # Used in train/val/test
        self.phase_iou = torchmetrics.JaccardIndex(num_classes=5,task="multiclass")
        self.phase_iou_per_class = torchmetrics.JaccardIndex(num_classes=5, task="multiclass", average=None)
        self.mask_iou = torchmetrics.JaccardIndex(num_classes=2,task="multiclass")
        self.mask_iou_per_class = torchmetrics.JaccardIndex(num_classes=2, task="multiclass", average=None)

        # Define meanmetrics, used to avg loss across gpus/batches during training/val steps
        self.train_loss_avg = torchmetrics.MeanMetric()
        self.val_loss_avg = torchmetrics.MeanMetric()
        self.test_loss_avg = torchmetrics.MeanMetric()
        # We track phase prediction iou
        self.train_pp_iou_avg = torchmetrics.MeanMetric()
        self.val_pp_iou_avg = torchmetrics.MeanMetric()
        self.test_pp_iou_avg = torchmetrics.MeanMetric()
        # And cloud mask iou
        self.train_cm_iou_avg = torchmetrics.MeanMetric()
        self.val_cm_iou_avg = torchmetrics.MeanMetric()
        self.test_cm_iou_avg = torchmetrics.MeanMetric()

        self.r2_score_cod = R2Score()
        self.r2_score_cps = R2Score()

        
    def forward(self,x):
        x = self.preprocessor(x)
        shared_feats = self.unet(x) # B 256 128 128. This passes x through satvis and unet
        #print("shared_feats shape", shared_feats.shape)
        mask_logits = self.mask_head(shared_feats) # B 1 128 128

        phase_and_mask = torch.cat([shared_feats,mask_logits],dim=1)
        cod_and_mask = torch.cat([shared_feats,mask_logits],dim=1)
        cps_and_mask = torch.cat([shared_feats,mask_logits],dim=1)


        phase_logits = self.phase_head(phase_and_mask) # B 5 128 128
        cod_output = self.cod_head(cod_and_mask) # B 1 128 128
        cps_output = self.cps_head(cps_and_mask)  # same as above

        return {
            'mask_logits': mask_logits,
            'phase_logits': phase_logits,
            'cps_output': cps_output,
            'cod_output': cod_output
        }


    def forward_old(self, x):
        input_size = x.shape[-2:]
        x = self.preprocessor(x)
        feats = self.encoder(x)
        decoded_m = self.decoder_m(feats)
        decoded_p = self.decoder_p(feats)
        decoded_cod = self.decoder_cod(feats)
        decoded_cps = self.decoder_cps(feats)



        mask_logits = self.mask_head(decoded_m)
        mask_logits = F.interpolate(mask_logits, size=input_size, mode="bilinear", align_corners=False)
        mask_probs = torch.sigmoid(mask_logits)  # shape [B, 1, H, W]

        # Verify binary mask is working as it should - class 1 is cloudy so take 1:2 where the prob was >= 0.5 gives u the mask (1 for cloudy so not cloudy is 0s)
        binary_mask = (mask_probs >= 0.5).float() # shape b,1,h,w
        binary_mask = binary_mask[:,1:2,:,:]
        #TODO I don't want forward method getting so long but having four decoder variables is getting ANNOYING right now the code will error here unless all decoder have the same shape. I'll fix this tomorrow (thurs)
        binary_mask_decoder = binary_mask.expand(-1, decoded_p.size(1), -1, -1)
        binary_mask_decoder = binary_mask_decoder.to(decoded_p.dtype)
        #print("binary_mask_decoder shape", binary_mask_decoder.shape)

        binary_mask_decoder_2 = binary_mask.expand(-1,decoded_cod.size(1),-1,-1)
        binary_mask_decoder_2 = binary_mask_decoder_2.to(decoded_cod.dtype)
        #features in non cloudy regions become zero
        x_cloudy_phase = decoded_p * binary_mask_decoder
        x_cloudy_cod = decoded_cod * binary_mask_decoder_2
        x_cloudy_cps = decoded_cps * binary_mask_decoder_2


        phase_logits = self.phase_head(x_cloudy_phase)
        phase_logits = F.interpolate(phase_logits, size=input_size, mode="bilinear", align_corners=False)
        # Apply the cloud mask condition: set phase logits to 0 where cloud mask is 0
        final_phase_logits = torch.clone(phase_logits)

        not_cloudy_pixel_mask = (binary_mask == 0).bool().squeeze(1) # b h w
        #print("not_cloudy_pixel_mask shape", not_cloudy_pixel_mask.shape)

        ### HANDLE LOGITS FOR CLASS 0 (NOT CLOUDY) 
        ### If not_cloudy_mask is True it will set class 0 logit to 0 (neutral/slightly pos value to favor it)
        ### otherwise, keeps original class 0 logit
        final_phase_logits[:, 0, :, :] = torch.where(
            not_cloudy_pixel_mask.squeeze(1), # Condition for the pixel (squeezed to match 2D slice)
            torch.full_like(phase_logits[:, 0, :, :], 0.0), # Value if not cloudy (explicitly shaped)
            phase_logits[:, 0, :, :] # Value if cloudy (original logit)
        )
    
        ### this basically just makes the not cloudy mask be expanded to apply it to the other four logits
        expanded_not_cloudy_mask_for_other_classes = not_cloudy_pixel_mask.unsqueeze(1).expand(
                -1, self.num_phase_classes - 1, -1, -1
                )

        ### now handle the other logits! note the indexing 1: because we alr handled the logit for class 0 
        ### if not cloudy we set the logits for the other class to a small engative value (so it'll prefer class 0)
        final_phase_logits[:, 1:, :, :] = torch.where(
                expanded_not_cloudy_mask_for_other_classes,
                torch.full_like(phase_logits[:,1:,:,:], -1e9),
                phase_logits[:,1:,:,:]
                )

        cps_output = self.cps_head(x_cloudy_cps)
        cod_output = self.cod_head(x_cloudy_cod)

        ### MAKE REGRESSION OUTPUTS 0 FOR PIXELS PREDICTED TO BE NOT CLOUDY ###
        expanded_not_cloudy_mask_for_regression = not_cloudy_pixel_mask.unsqueeze(1).expand(
            -1, 1, -1, -1
        )

        final_cps_output = torch.where(
            expanded_not_cloudy_mask_for_regression,
            torch.full_like(cps_output, 0.0), # Set to 0.0 if not cloudy
            cps_output # Keep original value if cloudy
        )

        final_cod_output = torch.where(
                expanded_not_cloudy_mask_for_regression,
                torch.full_like(cod_output, 0.0),
                cod_output
                )

        return {
            'mask_logits': mask_logits,
            'phase_logits': final_phase_logits,
            'cps_output': final_cps_output,
            'cod_output': final_cod_output
        }


    def training_step(self, batch, batch_idx):

        inputs, targets = batch
        #print(inputs.shape)

        outputs = self.forward(inputs)  


        phase_logits = outputs['phase_logits']


        # Note that I pass in OG phase logits to loss below rn, but i'll leave this code here
        cloudy_pixel_mask = (targets['mask'] == 1).squeeze(1) # B H W
        target_phase_labels_cloudy = targets["phase"][cloudy_pixel_mask].long()
        final_phase_logits_flat = phase_logits.permute(0, 2, 3, 1).reshape(-1, self.num_phase_classes)
        # Reshape mask to [B*H*W]
        cloudy_pixel_mask_flat = cloudy_pixel_mask.reshape(-1)
        # Select logits corresponding to cloudy pixels
        logits_cloudy = final_phase_logits_flat[cloudy_pixel_mask_flat]


        # ====== COMPUTE LOSSES
        mask_loss = self.mask_loss(outputs['mask_logits'], targets['mask'])  # Binary segmentation
        
        ce = self.ce_loss(phase_logits,targets['phase'].long())
        dice = self.dice_loss(phase_logits,targets['phase'].long())
        phase_loss = self.dice_weight * dice + (1 - self.dice_weight) * ce

        cps_loss = self.cps_loss(outputs['cps_output'].squeeze(1), targets['cps'])  # Regression loss
        cod_loss = self.cod_loss(outputs['cod_output'].squeeze(1), targets['cod'])  # Regression loss
        task_losses = [mask_loss,phase_loss,cps_loss,cod_loss]
        
        #total_loss = self._calculate_total_uncertainty_loss(task_losses)

        total_loss = (
            self.mask_weight * mask_loss +
            self.phase_weight * phase_loss +
            self.cps_weight * cps_loss +
            self.cod_weight * cod_loss
            )
        
        # Phase Prediction IOU
        phase_preds = torch.argmax(logits_cloudy, dim=1)
        phase_iou = self.phase_iou(phase_preds, target_phase_labels_cloudy)
        self.train_pp_iou_avg.update(phase_iou)
        # per class
        phase_iou_per_class = self.phase_iou_per_class(phase_preds, target_phase_labels_cloudy)
        for i, class_iou in enumerate(phase_iou_per_class):
            self.log(f'train_phase_iou_{i}', class_iou, on_step=False, on_epoch=True)

        # Cloud Mask IOU
        mask_preds = torch.argmax(outputs['mask_logits'], dim=1)
        mask_iou = self.mask_iou(mask_preds, targets['mask'])
        self.train_cm_iou_avg.update(mask_iou)
        # per class
        mask_iou_per_class = self.mask_iou_per_class(mask_preds, targets["mask"])
        for i, class_iou in enumerate(mask_iou_per_class):
            self.log(f'train_mask_iou_{i}', class_iou, on_step=False, on_epoch=True)

        self.train_loss_avg.update(total_loss)
        self.log('train_loss', self.train_loss_avg.compute(),on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_mask_iou', self.train_cm_iou_avg.compute(),on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_phase_iou', self.train_pp_iou_avg.compute(),on_step=False, on_epoch=True, prog_bar=False)


        #self.log('mask loss weight', self.mask_weight,on_step=False, on_epoch=True)
        #self.log('phase loss weight', self.phase_weight, on_step=False, on_epoch = True)
        #self.log('cps loss weight', self.cps_weight, on_step = False, on_epoch = True)
        #self.log('cod loss weight', self.cod_weight, on_step = False, on_epoch = True)

        return total_loss

    def validation_step(self, batch, batch_idx):

        inputs, targets = batch

        outputs = self.forward(inputs)  
        
        mask_loss = self.mask_loss(outputs['mask_logits'], targets['mask'])  # Binary segmentation
        
        ce = self.ce_loss(outputs['phase_logits'],targets['phase'].long())
        dice = self.dice_loss(outputs['phase_logits'],targets['phase'].long())
        phase_loss = self.dice_weight * dice + (1 - self.dice_weight) * ce

        cps_loss = self.cps_loss(outputs['cps_output'].squeeze(1), targets['cps'])  # Regression loss
        cod_loss = self.cod_loss(outputs['cod_output'].squeeze(1), targets['cod'])  # Regression loss
        task_losses = [mask_loss,phase_loss,cps_loss,cod_loss]

        #total_loss = self._calculate_total_uncertainty_loss(task_losses)

        total_loss = (
            self.mask_weight * mask_loss +
            self.phase_weight * phase_loss +
            self.cps_weight * cps_loss +
            self.cod_weight * cod_loss
            )
        
        ### Cloud Phase IOU
        phase_preds = torch.argmax(outputs['phase_logits'], dim=1)
        phase_iou = self.phase_iou(phase_preds, targets['phase'])
        self.val_pp_iou_avg.update(phase_iou)
        # per class
        phase_iou_per_class = self.phase_iou_per_class(phase_preds, targets["phase"])
        for i, class_iou in enumerate(phase_iou_per_class):
            self.log(f'val_phase_iou_{i}', class_iou, on_step=False, on_epoch=True)

        ### Cloud Mask IOU
        mask_preds = torch.argmax(outputs['mask_logits'], dim=1)
        mask_iou = self.mask_iou(mask_preds, targets['mask'])
        self.val_cm_iou_avg.update(mask_iou)
        # per class
        mask_iou_per_class = self.mask_iou_per_class(mask_preds, targets["mask"])
        for i, class_iou in enumerate(mask_iou_per_class):
            self.log(f'val_mask_iou_{i}', class_iou, on_step=False, on_epoch=True)


        ### VISUALIZE 
        if self.current_epoch % 5 == 0:
            mask_preds = mask_preds.cpu()
            mask_targets = targets["mask"].cpu()
            phase_preds = phase_preds.cpu()
            phase_targets = targets["phase"].cpu()
            self.visualize(mask_preds,mask_targets,phase_preds,phase_targets,batch_idx)



        ### Update loss and logs
        self.val_loss_avg.update(total_loss)
        self.log('val_loss', self.val_loss_avg.compute(),on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_mask_iou', self.val_cm_iou_avg.compute(),on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_phase_iou', self.val_pp_iou_avg.compute(),on_step=False, on_epoch=True, prog_bar=True)

        return total_loss
    
    def test_step(self, batch, batch_idx):

        inputs, targets = batch

        outputs = self.forward(inputs)  
        
        mask_loss = self.mask_loss(outputs['mask_logits'], targets['mask'])  # Binary segmentation
        
        ce = self.ce_loss(outputs['phase_logits'],targets['phase'].long())
        dice = self.dice_loss(outputs['phase_logits'],targets['phase'].long())
        phase_loss = self.dice_weight * dice + (1 - self.dice_weight) * ce

        cps_loss = self.cps_loss(outputs['cps_output'].squeeze(1), targets['cps'])  # Regression loss
        cod_loss = self.cod_loss(outputs['cod_output'].squeeze(1), targets['cod'])  # Regression loss
        task_losses = [mask_loss,phase_loss,cps_loss,cod_loss]

        #total_loss = self._calculate_total_uncertainty_loss(task_losses)

        total_loss = (
            self.mask_weight * mask_loss +
            self.phase_weight * phase_loss +
            self.cps_weight * cps_loss +
            self.cod_weight * cod_loss
            )

        # Phase Prediction IOU
        phase_preds = torch.argmax(outputs['phase_logits'], dim=1)
        phase_iou = self.phase_iou(phase_preds, targets['phase'])
        self.test_pp_iou_avg.update(phase_iou)
        # per class
        phase_iou_per_class = self.phase_iou_per_class(phase_preds, targets["phase"])
        for i, class_iou in enumerate(phase_iou_per_class):
            self.log(f'test_phase_iou_{i}', class_iou, on_step=False, on_epoch=True)
        # update list of all predictions and targets (used for classification report, see on_test_epoch_end)
        self.all_preds_phase.append(phase_preds.flatten().cpu().numpy())
        self.all_targets_phase.append(targets["phase"].flatten().cpu().numpy())


        # Cloud Mask IOU
        mask_preds = torch.argmax(outputs['mask_logits'], dim=1)
        mask_iou = self.mask_iou(mask_preds, targets['mask'])
        self.test_cm_iou_avg.update(mask_iou)
        # per class
        mask_iou_per_class = self.mask_iou_per_class(mask_preds, targets["mask"])
        for i, class_iou in enumerate(mask_iou_per_class):
            self.log(f'test_mask_iou_{i}', class_iou, on_step=False, on_epoch=True)
        # update list of all predictions and targets (used for classification report, see on_test_epoch_end)
        self.all_preds_mask.append(mask_preds.flatten().cpu().numpy())
        self.all_targets_mask.append(targets["mask"].flatten().cpu().numpy())

        # VISUALIZE !!
        
        mask_preds = mask_preds.cpu()
        mask_targets = targets["mask"].cpu()
        phase_preds = phase_preds.cpu()
        phase_targets = targets["phase"].cpu()
        self.visualize(mask_preds,mask_targets,phase_preds,phase_targets,batch_idx)


        # Update R2 score for each regression task
        # cod
        preds_cod = outputs['cod_output']
        targets_cod = targets["cod"].unsqueeze(1)

        preds_flat_cod = preds_cod.view(preds_cod.size(0), -1)
        targets_flat_cod = targets_cod.view(targets_cod.size(0), -1)
        self.r2_score_cod.update(preds_flat_cod, targets_flat_cod)

        # cps
        preds_cps = outputs['cps_output']
        targets_cps = targets["cps"].unsqueeze(1)

        preds_flat_cps = preds_cps.view(preds_cps.size(0), -1)
        targets_flat_cps = targets_cps.view(targets_cps.size(0), -1)
        self.r2_score_cps.update(preds_flat_cps, targets_flat_cps)

        # Update loss avg and log iou
        self.test_loss_avg.update(total_loss)
        self.log('test_mask_iou', self.test_cm_iou_avg.compute(),on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_phase_iou', self.test_pp_iou_avg.compute(),on_step=False, on_epoch=True, prog_bar=True)

        return total_loss

    def on_test_epoch_end(self):
        # Flatten the lists of predictions and targets
        all_preds_phase = np.concatenate(self.all_preds_phase)
        all_targets_phase = np.concatenate(self.all_targets_phase)

        # Compute the classification report
        report_phase = classification_report(all_targets_phase, all_preds_phase, output_dict=True)

        print("===== CLASSIFICATION REPORT: CLOUD PHASE =====")
        print(report_phase)

        all_preds_mask = np.concatenate(self.all_preds_mask)
        all_targets_mask = np.concatenate(self.all_targets_mask)

        # Compute the classification report
        report_mask= classification_report(all_targets_mask, all_preds_mask, output_dict=True)

        # Log the full classification report or individual metrics
        print("====== CLASSIFICATION REPORT: CLOUD MASK =====")
        print(report_mask)

        self.log("test_r2_cod", self.r2_score_cod.compute(), prog_bar=True)
        self.log("test_r2_cps", self.r2_score_cps.compute(), prog_bar=True)

    def visualize(self,mask_preds,mask_targets,phase_preds,phase_targets,batch_idx):
        # Visualize cloud mask predictions
        os.makedirs("MultitaskVisuals", exist_ok=True)

        mask_spec = [
                    (0, '#3498db', 'No Cloud'),  # Blue
                    (1, '#e74c3c', 'Cloud'),     # Red
                    ]

        cmap_mask = plt.cm.colors.ListedColormap([c[1] for c in mask_spec])
        tick_vals = [c[0] for c in mask_spec]
        tick_labels = [c[2] for c in mask_spec]

        for i in range(min(3, mask_preds.size(0))):  # visualize 2 examples max
            pred_mask = mask_preds[i]
            truth_mask = mask_targets[i]

            fig, axs = plt.subplots(1,2, figsize=(10,4))

            im0 = axs[0].imshow(pred_mask, cmap=cmap_mask,vmin=-0.5,vmax=5.5)
            axs[0].set_title("Prediction")
            axs[0].axis("off")

            im1 = axs[1].imshow(truth_mask, cmap=cmap_mask, vmin=-0.5, vmax=5.5)
            axs[1].set_title("Ground Truth")
            axs[1].axis("off")

            cbar = fig.colorbar(im1, ax=axs.ravel().tolist(), ticks=tick_vals, shrink=0.8,fraction=0.046, pad=0.04)
            cbar.set_ticks(tick_vals)
            cbar.ax.set_yticklabels(tick_labels, fontsize=8)

            fig.suptitle(f"Epoch {self.current_epoch} | Batch {batch_idx} | Sample {i}", fontsize=12)
            #plt.tight_layout()

                # get job id, or if running locally outputs "unknown job"
            job_id = os.environ.get("SLURM_JOB_ID", "unknown_job")
            save_dir = os.path.join("MultitaskVisuals",f"job_{job_id}","CloudMask")
            os.makedirs(save_dir, exist_ok=True)

            plt.savefig(os.path.join(save_dir, f"epoch{self.current_epoch}_batch{batch_idx}_example{i}.png"))
            plt.close()

        ##### CLOUD PHASE
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

        for i in range(min(3, phase_preds.size(0))):  # visualize 2 examples max
            pred_mask = phase_preds[i]
            truth_mask = phase_targets[i]

            fig, axs = plt.subplots(1,2, figsize=(10,4))

            im0 = axs[0].imshow(pred_mask, cmap=cmap_phase,vmin=-0.5,vmax=4.5)
            axs[0].set_title("Prediction")
            axs[0].axis("off")

            im1 = axs[1].imshow(truth_mask, cmap=cmap_phase, vmin=-0.5, vmax=4.5)
            axs[1].set_title("Ground Truth")
            axs[1].axis("off")

            cbar = fig.colorbar(im1, ax=axs.ravel().tolist(), orientation='vertical',
                        ticks=tick_vals, fraction=0.1, pad=0.1)
            cbar.set_ticks(tick_vals)
            cbar.ax.set_yticklabels(tick_labels, fontsize=8)

            fig.suptitle(f"Test Predictions Sample {i}", fontsize=12)
            #plt.tight_layout()

            # get job id, or if running locally outputs "unknown job"
            job_id = os.environ.get("SLURM_JOB_ID", "unknown_job")
            save_dir = os.path.join("MultitaskVisuals",f"job_{job_id}","CloudPhase")
            os.makedirs(save_dir, exist_ok=True)

            plt.savefig(os.path.join(save_dir, f"example{i}.png"))
            plt.close()



    def configure_optimizers(self):
        optimizer = build_optimizer(self.config, self, is_pretrain=False)
        print(f'Using optimizer: {optimizer}')

        return optimizer

    def on_train_epoch_start(self):
        self.train_loss_avg.reset()
        self.train_pp_iou_avg.reset()
        self.train_cm_iou_avg.reset()
        

    def on_validation_epoch_start(self):
        self.val_loss_avg.reset()
        self.val_pp_iou_avg.reset()
        self.val_cm_iou_avg.reset()
                                                       

    # === methods for optimizer configuration ===
    def no_weight_decay(self):
        """
        Returns a set of parameter names that should NOT have weight decay.
        Usually includes bias terms and normalization layer parameters.
        """
        no_decay_params = set()
        for name, param in self.named_parameters():
            # Example for standard bias and LayerNorm/BatchNorm params
            if 'bias' in name or 'norm' in name:
                no_decay_params.add(name)
        # Add your log_vars explicitly
        no_decay_params.add('log_vars') # Ensure log_vars are not decayed
        return no_decay_params

    def no_weight_decay_keywords(self):
        """
        Returns a set of keywords for parameter names that should NOT have weight decay.
        Used when you can't list exact names but patterns.
        """
        # This can be used in conjunction with or instead of no_weight_decay()
        # Your existing `no_weight_decay_keywords` in the build_optimizer
        # likely processes these.
        return {'bias', 'norm'} # Example common keywords
