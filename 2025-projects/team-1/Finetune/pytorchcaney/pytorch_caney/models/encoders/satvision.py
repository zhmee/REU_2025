from .swinv2modified import SwinTransformerV2, PromptedSwinTransformerV2
from ..model_factory import ModelFactory
import torch.nn as nn
import torch


# -----------------------------------------------------------------------------
# SatVision
# -----------------------------------------------------------------------------
@ModelFactory.encoder("satvision")
class SatVision(nn.Module):

    # -------------------------------------------------------------------------
    # __init__
    # -------------------------------------------------------------------------
    def __init__(self, config):
        super().__init__()

        self.config = config

        window_sizes = config.MODEL.SWINV2.PRETRAINED_WINDOW_SIZES
        
        if self.config.MODEL.VPT == False:
            self.model = SwinTransformerV2(
                img_size=config.DATA.IMG_SIZE,
                patch_size=config.MODEL.SWINV2.PATCH_SIZE,
                in_chans=config.MODEL.SWINV2.IN_CHANS,
                num_classes=config.MODEL.NUM_CLASSES,
                embed_dim=config.MODEL.SWINV2.EMBED_DIM,
                depths=config.MODEL.SWINV2.DEPTHS,
                num_heads=config.MODEL.SWINV2.NUM_HEADS,
                window_size=config.MODEL.SWINV2.WINDOW_SIZE,
                mlp_ratio=config.MODEL.SWINV2.MLP_RATIO,
                qkv_bias=config.MODEL.SWINV2.QKV_BIAS,
                drop_rate=config.MODEL.DROP_RATE,
                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                ape=config.MODEL.SWINV2.APE,
                patch_norm=config.MODEL.SWINV2.PATCH_NORM,
                use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                pretrained_window_sizes=window_sizes,
                )
        else:
            print("Using prompted swin transformer")
            self.model = PromptedSwinTransformerV2(
                img_size=config.DATA.IMG_SIZE,
                patch_size=config.MODEL.SWINV2.PATCH_SIZE,
                in_chans=config.MODEL.SWINV2.IN_CHANS,
                num_classes=config.MODEL.NUM_CLASSES,
                embed_dim=config.MODEL.SWINV2.EMBED_DIM,
                depths=config.MODEL.SWINV2.DEPTHS,
                num_heads=config.MODEL.SWINV2.NUM_HEADS,
                window_size=config.MODEL.SWINV2.WINDOW_SIZE,
                mlp_ratio=config.MODEL.SWINV2.MLP_RATIO,
                qkv_bias=config.MODEL.SWINV2.QKV_BIAS,
                drop_rate=config.MODEL.DROP_RATE,
                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                ape=config.MODEL.SWINV2.APE,
                patch_norm=config.MODEL.SWINV2.PATCH_NORM,
                use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                pretrained_window_sizes=window_sizes,
                prompt_length = config.MODEL.NUM_PROMPTS)
            # Freeze all parameters except prompt
            # Freeze everything
            for param in self.model.parameters():
                param.requires_grad = False

            # Unfreeze the prompt embeddings
            self.model.prompt_embeddings.requires_grad = True

        if self.config.MODEL.PRETRAINED:
            self.load_pretrained()

        self.num_classes = self.model.num_classes
        self.num_layers = self.model.num_layers
        self.num_features = self.model.num_features

    # -------------------------------------------------------------------------
    # __init__
    # -------------------------------------------------------------------------
    def load_pretrained(self):
        print("load_pretrained() called!")

        checkpoint = torch.load(
            self.config.MODEL.PRETRAINED, map_location='cpu')

        checkpoint_model = checkpoint['module']

        if any([True if 'encoder.' in k else
                False for k in checkpoint_model.keys()]):

            
            checkpoint_model = {k.replace(
                'encoder.', ''): v for k, v in checkpoint_model.items()
                if k.startswith('encoder.')}

            print('Detect pre-trained model, remove [encoder.] prefix.')

        else:

            print(
                'Detect non-pre-trained model, pass without doing anything.')


        #old_weight = checkpoint_model['patch_embed.proj.weight']  # [out_channels, 14, kernel_h, kernel_w]
        #if old_weight.shape[1] == 14 and self.config.MODEL.SWINV2.IN_CHANS == 16:
        #    new_weight = torch.zeros((old_weight.shape[0], 16, *old_weight.shape[2:]))
        #    new_weight[:, :14] = old_weight
        #    checkpoint_model['patch_embed.proj.weight'] = new_weight
        #    print("Zero-padded input conv weights from 14 → 16 bands.")
        #else: 
        #    print("No zero padding")
        
        msg = self.model.load_state_dict(checkpoint_model, strict=False)
    
        print("After loading, patch_embed.proj.weight.shape:",self.model.patch_embed.proj.weight.shape)

        print(msg)

        del checkpoint

        torch.cuda.empty_cache()

        print(f">>>>>>> loaded successfully '{self.config.MODEL.PRETRAINED}'")

    # -------------------------------------------------------------------------
    # forward
    # -------------------------------------------------------------------------
    def forward(self, x):
        return self.model.forward(x)

    # -------------------------------------------------------------------------
    # forward_features
    # -------------------------------------------------------------------------
    def forward_features(self, x):
        return self.model.forward_features(x)

    # -------------------------------------------------------------------------
    # extra_features
    # -------------------------------------------------------------------------
    def extra_features(self, x):
        return self.model.extra_features(x)
