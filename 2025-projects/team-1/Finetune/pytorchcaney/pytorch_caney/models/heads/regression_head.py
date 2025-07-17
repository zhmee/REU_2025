import torch
import torch.nn as nn
from ..model_factory import ModelFactory


#@ModelFactory.head("regression_head")
#class RegressionHead(nn.Module):
#    def __init__(self, decoder_channels=64, output_dim=1, head_dropout=0.1):
#        super().__init__()
#        output_shape = (128,128)
#
#        # Flatten the output from the decoder
#        self.head = nn.Sequential(
#            nn.Flatten(),  # Flatten the feature map from the decoder
#            nn.Linear(decoder_channels * 128 * 128, 512),  # First fully connected layer
#            nn.ReLU(inplace=True),  # Activation
#            nn.Dropout(head_dropout),
#            nn.Linear(512, output_dim),  # Output layer for regression
#        )
#
#    def forward(self, x):
#        return self.head(x)

@ModelFactory.head("regression_head_MLP")
class RegressionHeadMLP(nn.Module):
    def __init__(self, decoder_channels=64, output_dim=1, head_dropout=0.1):
        super().__init__()

        self.flatten = nn.Flatten(start_dim=1) # Flatten b,c,h,w to b, c*h*w
        self.mlp = nn.Sequential(
                nn.Linear(decoder_channels * 128 * 128, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512,1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024,128*128)
                )


    def forward(self, x):
        x = self.flatten(x)
        x = self.mlp(x)
        x = x.view(-1,1,128,128)
        return x



@ModelFactory.head("regression_head")
class RegressionHead(nn.Module):
    def __init__(self, decoder_channels=64, output_dim=1, head_dropout=0.1):
        super().__init__()
        output_shape = (128,128)
        self.head = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, padding=1), 
            nn.Dropout(head_dropout),
            nn.Upsample(size=output_shape, mode='bilinear', align_corners=False)  # Upsample to output shape
        )
    def forward(self,x):
        #print("x shape before fwd in regression head", x.shape)
        return self.head(x)
