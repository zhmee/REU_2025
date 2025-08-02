import lightning as pl
import torch
import torch.nn as nn
import time
import numpy as np
import utils
from torchmetrics import Accuracy

### 2025 NEW EVENT CLASSIFIER TRANSFORMER ###
class EventClassifierTransformerNetwork(pl.LightningModule):
    """Event Classifier Transformer for 13-class classification, built with PyTorch Lightning.

    This model adapts the transformer architecture to handle tabular data by treating
    each feature as an individual token. It uses a transformer encoder to process the
    feature tokens and a special class token to aggregate information for the final
    multi-class prediction. It also incorporates the specified custom loss function.
    """
    def __init__(self,
                 indim: int, outdim: int, nodes: int, nheads: int, linear_factor: int, lr: float,
                 l2: float, dropout: float, num_layers: int,  penalty: float, optimizer: str, lr_step: int,
                 lr_gam: float, custom_loss: bool = True, activation: str = 'relu'):
        """
        Args:
            indim (int): Number of input features (n_vars).
            outdim (int): Number of output classes.
            nodes (int): The embedding dimension for each feature.
            nheads (int): Number of heads for multi-head attention. Must be a divisor of `nodes`.
            linear_factor (int): Multiplicative factor for the hidden dimension in feed-forward layers.
            lr (float): Learning rate.
            l2 (float): L2 regularization (weight decay).
            dropout (float): Dropout rate.
            penalty (float): Penalty coefficient for the custom loss.
            optimizer (str): Name of the optimizer to use (e.g., 'adamw').
            lr_step (int): Step size for the learning rate scheduler.
            lr_gam (float): Gamma factor for the learning rate scheduler.
            custom_loss (bool): Whether to use the custom loss function.
        """
        super().__init__()
        self.save_hyperparameters()

        # Metrics
        self.train_acc = Accuracy(task='multiclass', num_classes=outdim)
        self.valid_acc = Accuracy(task='multiclass', num_classes=outdim)
        self.test_acc = Accuracy(task='multiclass', num_classes=outdim)
        self.validation_step_outputs = []
        if activation.lower() == 'relu':
            self.activation_fn = nn.ReLU()
        elif activation.lower() == 'gelu':
            self.activation_fn = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        # --- Architecture ---

        # 1. Feature Embedding: Each of the `indim` features is projected to an embedding of size `nodes`.
        self.embed = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, linear_factor * nodes),
                self.activation_fn,
                nn.Linear(linear_factor * nodes, nodes)
            ) for _ in range(indim)
        ])

        self.transformer_blocks = nn.ModuleList([
    nn.Sequential(
        nn.LayerNorm(nodes),
        nn.MultiheadAttention(embed_dim=nodes, num_heads=nheads, dropout=dropout, batch_first=True),
        nn.LayerNorm(nodes),
        nn.Sequential(
            nn.Linear(nodes, linear_factor * nodes),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(linear_factor * nodes, nodes),
            nn.Dropout(dropout)
        )
    )
    for _ in range(num_layers)
])

        # 3. Class (CLS) Token and Classifier Head
        self.cls_token = nn.Parameter(torch.zeros(1, 1, nodes), requires_grad=True)
        self.cls_pre_attn_norm = nn.LayerNorm(nodes)
        self.cls_attn = nn.MultiheadAttention(embed_dim=nodes, num_heads=nheads, dropout=dropout, batch_first=True)
        self.cls_pre_fc_norm = nn.LayerNorm(nodes)
        self.cls_fc = nn.Linear(nodes, outdim)

        self.dropout_layer = nn.Dropout(dropout)
        self.optimizer_pointer = utils.get_optimizer(optimizer)(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.l2)
        self.start_time = time.time()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the transformer.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_features).
        Returns:
            torch.Tensor: Logits of shape (batch_size, n_classes).
        """
        # Embed each feature: [batch, n_features] -> [batch, n_features, nodes]
        var_embed_list = [self.embed[i](x[:, i].unsqueeze(1)) for i in range(self.hparams.indim)]

        x = torch.stack(var_embed_list, axis=1)
        for block in self.transformer_blocks:
            residual = x
            x = block[0](x)
            x_attn = block[1](x, x, x)[0]
            x = residual + self.dropout_layer(x_attn)
            residual = x
            x = block[2](x)
            x = residual + block[3](x)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        u = torch.cat((cls_tokens, x), dim=1)

        # CLS token attention
        residual = cls_tokens
        u_norm = self.cls_pre_attn_norm(u)
        cls_attn_out = self.cls_attn(u_norm[:, :1, :], u_norm, u_norm)[0]
        x = residual + self.dropout_layer(cls_attn_out)

        # Final Classifier
        x = self.cls_pre_fc_norm(x)
        x = x.flatten(start_dim=1)
        logits = self.cls_fc(x)
        return logits

    def loss_penalty(self, pred_one_hot: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
        assert targ.min() >= 0 and targ.max() < self.hparams.outdim, \
    f"Invalid label values: min={targ.min().item()}, max={targ.max().item()}"

        D = torch.tensor([
            [1, 2, 2, 2, 2, 2, 10, 10, 10, 10, 10, 10, 20],
            [2, 1, 2, 2, 2, 2, 10, 10, 10, 10, 10, 10, 20],
            [2, 2, 1, 2, 2, 2, 10, 10, 10, 10, 10, 10, 20],
            [2, 2, 2, 1, 2, 2, 10, 10, 10, 10, 10, 10, 20],
            [2, 2, 2, 2, 1, 2, 10, 10, 10, 10, 10, 10, 20],
            [2, 2, 2, 2, 2, 1, 10, 10, 10, 10, 10, 10, 20],
            [10, 10, 10, 10, 10, 10, 1, 8, 2, 2, 2, 2, 20],
            [10, 10, 10, 10, 10, 10, 10, 1, 8, 2, 2, 2, 20],
            [10, 10, 10, 10, 10, 8, 2, 8, 1, 8, 2, 2, 20],
            [10, 10, 10, 10, 10, 10, 2, 2, 8, 1, 8, 2, 20],
            [10, 10, 10, 10, 10, 10, 2, 2, 2, 8, 1, 8, 20],
            [10, 10, 10, 10, 10, 10, 2, 2, 2, 2, 8, 1, 20],
            [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 1]
        ], dtype=torch.float32, device=self.device)
        D -= D * torch.eye(self.hparams.outdim, device=self.device)
        return (D[targ] * pred_one_hot).mean()

      def loss_fn(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cross_entropy = nn.functional.cross_entropy(logits, y)
        if not self.hparams.custom_loss:
            return cross_entropy

        # Create one-hot encoding of predictions for the penalty term
        preds = torch.argmax(logits, dim=1)
        pred_one_hot = nn.functional.one_hot(preds, num_classes=self.hparams.outdim).float()

        pair_wise_penalty = self.loss_penalty(pred_one_hot, y)
        return cross_entropy * (1 + pair_wise_penalty) ** self.hparams.penalty

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        self.log('lr', self.optimizer_pointer.state_dict()['param_groups'][0]['lr'], sync_dist=True) 
        self.log("time", time.time() - self.start_time, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.valid_acc(preds, y)
        self.log("valid_loss", loss, prog_bar=True, sync_dist=True)
        self.log("valid_acc", self.valid_acc, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_acc(preds, y)
        self.log("test_loss", loss, sync_dist=True)
        self.log("test_acc", self.test_acc, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = self.optimizer_pointer
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.hparams.lr_step, gamma=self.hparams.lr_gam)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        return preds

# Lightning implementation of REU 2023's best achieved model
class LSTM2023(pl.LightningModule):
    def __init__(self, indim, outdim, num_layers, neurons, lr, lr_step, lr_gam, dropout, activation, optimizer, l2=0, log_test=False):
        super().__init__()
        self.save_hyperparameters()

        # PL attributes
        self.train_acc = Accuracy(task='multiclass', num_classes=13)
        self.valid_acc = Accuracy(task='multiclass', num_classes=13)
        self.test_acc = Accuracy(task='multiclass', num_classes=13)

        self.indim = indim
        self.outdim = outdim
        self.num_layers = num_layers
        self.neurons = neurons
        self.lr = lr
        self.lr_step = lr_step
        self.lr_gam = lr_gam
        self.l2=l2

        self.input_layer = nn.Linear(self.indim, self.neurons)
        self.lstm = nn.LSTM(input_size=self.neurons, hidden_size=self.neurons, num_layers=4, batch_first=True)
        self.rnn_layers = self.get_layers()  # dense layers
        self.dropout = nn.Dropout(dropout)
        self.activation = utils.get_activation(activation)
        self.output_layer = nn.Linear(self.neurons, self.outdim)

        self.optimizer = utils.get_optimizer(optimizer)(self.parameters(), lr=lr, weight_decay=self.l2)
        self.start_time = time.time()
        self.log_test = log_test

    def forward(self, x, tgt, dropout_on=True):
        # x, tgt = batch
        x = x.unsqueeze(1)  # Adds a sequence length of 1
        x = self.input_layer(x)
        
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(4, x.size(0), self.neurons).to(x.device).requires_grad_()
        # Initializing cell state for first input with zeros
        c0 = torch.zeros(4, x.size(0), self.neurons).to(x.device).requires_grad_()
        # Forward propagation by passing in the input, hidden state, and cell state into the model
        x, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        x = x[:, -1, :]

        # RNN layer
        for layer in self.rnn_layers: 
            if isinstance(layer, nn.Linear):
                x = layer(x)
                # Dropout
                if dropout_on:
                    x = self.dropout(x) # only want the final timestep output
            else:
                x, tgt = layer(x, tgt)
        
        x = self.activation(x) 
        x = self.output_layer(x)
        return x

    def get_layers(self):
        layers = nn.ModuleList()
        layer = nn.Linear(self.neurons, self.neurons, bias=True)
        for _ in range(self.num_layers - 1):
            layers.append(layer)
        layers.append(layer)  
        return layers

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x, y)
        loss = nn.functional.cross_entropy(self(x, y), y.view(-1))
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, y)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def on_train_epoch_end(self):
        self.log('lr', self.optimizer.state_dict()['param_groups'][0]['lr'], sync_dist=True) 
        self.log("time", time.time() - self.start_time, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x, y, dropout_on=False)
        loss = nn.functional.cross_entropy(self(x, y), y.view(-1))
        preds = torch.argmax(logits, dim=1)
        self.valid_acc(preds, y)
        self.log("valid_acc", self.valid_acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("valid_loss", loss, prog_bar=True, sync_dist=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x, y, dropout_on=False)
        loss = nn.functional.cross_entropy(self(x, y), y.view(-1))
        preds = torch.argmax(logits, dim=1)
        self.test_acc(preds, y)
        self.log("test_acc", self.test_acc, prog_bar=True, sync_dist=True)  # let the hook determine the mode
        self.log("test_loss", loss, prog_bar=True, sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = self.optimizer
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_step, gamma=self.lr_gam)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch  # see predict.py, was running into an Attribute no Flatten error [resolved]
        logits = self(x, torch.empty(x.size(dim=0)), dropout_on=False)  # SUPER HACKY, CHECK THIS
        preds = torch.argmax(logits, dim=1)
        return preds

# FCN experimentation
class ImprovedFCN(pl.LightningModule):    
    def __init__(self, input_size, num_classes, hidden_layers, activation, lr, lr_step, lr_gam, penalty, dropout=0, l2=0):
        super().__init__()
        self.save_hyperparameters()

        # PL attributes
        self.train_acc = Accuracy(task='multiclass', num_classes=13)
        self.valid_acc = Accuracy(task='multiclass', num_classes=13)
        self.test_acc = Accuracy(task='multiclass', num_classes=13)

        # Parameters
        self.input_size = input_size
        self.num_classes = num_classes
        
        self.hidden_layers = hidden_layers
        self.activation = utils.get_activation(activation)
        self.dropout_layer = nn.Dropout(dropout)

        self.lr = lr
        self.lr_step = lr_step
        self.lr_gam = lr_gam
        self.l2 = l2

        self.penalty = penalty
        self.optimizer_pointer = None
        self.start_time = time.time()

        # FCN model
        self.output_layer = nn.Linear(self.hidden_layers[-1], num_classes)
        in_layer = input_size
        model_layers = [nn.Flatten()]
        for num in self.hidden_layers:
            layer = nn.Linear(in_layer, num)
            model_layers.append(layer)
            model_layers.append(self.activation)
            model_layers.append(self.dropout_layer)
            in_layer = num
        self.model = nn.Sequential(*model_layers)

    def forward(self, x):  # , dropout_on=True
        x = self.model(x)  
        x = self.output_layer(x)
        return x
    
    def loss_fn(self, x, y):
        # return (nn.functional.cross_entropy(X, y)) + self.misclassB(X, y)
        cross_entropy = nn.functional.cross_entropy(self(x), y) 

        arr = torch.argmax(self(x), dim=1)
        encoded_arr = torch.zeros((arr.size(dim=0), 13), dtype=torch.int).to(x.device)
        encoded_arr[torch.arange(arr.size(dim=0)), arr] = 1
        pair_wise_x = encoded_arr
        pair_wise = (1 + self.loss_penalty(pair_wise_x, y)) ** self.penalty  
        return cross_entropy * pair_wise

    # Give incorrect predictions a logarithmic divergence
    def misclassB (self, pred, targ):   # pred[nBatch, nClass], targ[nBatch], D[nClass, nClass]
        D = [ [1, 2, 2, 2, 2, 2, 8, 8, 8, 8, 8, 8, 20],
        [2, 1, 2, 2, 2, 2, 8, 8, 8, 8, 8, 8, 20],
        [2, 2, 1, 2, 2, 2, 8, 8, 8, 8, 8, 8, 20],
        [2, 2, 2, 1, 2, 2, 8, 8, 8, 8, 8, 8, 20],
        [2, 2, 2, 2, 1, 2, 8, 8, 8, 8, 8, 8, 20],
        [2, 2, 2, 2, 2, 1, 8, 8, 8, 8, 8, 8, 20],
        [8, 8, 8, 8, 8, 8, 1, 2, 2, 2, 2, 2, 20], 
        [8, 8, 8, 8, 8, 8, 2, 1, 2, 2, 2, 2, 20],
        [8, 8, 8, 8, 8, 8, 2, 2, 1, 2, 2, 2, 20],
        [8, 8, 8, 8, 8, 8, 2, 2, 2, 1, 2, 2, 20],
        [8, 8, 8, 8, 8, 8, 2, 2, 2, 2, 1, 2, 20],
        [8, 8, 8, 8, 8, 8, 2, 2, 2, 2, 2, 1, 20],
        [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 1] ]
        D = torch.from_numpy(np.asarray(D).astype(np.float32)).to(pred.device)
        D -= D * torch.eye(13).to(pred.device)
        return  (D[targ] * (-(pred - 1.0).clamp (min = 1.e-7).log())).mean()
    
    # Experimenting with loss penalty
    def loss_penalty (self, pred, targ): 
        D = [ [1, 2, 2, 2, 2, 2, 8, 8, 8, 8, 8, 8, 20],
        [2, 1, 2, 2, 2, 2, 8, 8, 8, 8, 8, 8, 20],
        [2, 2, 1, 2, 2, 2, 8, 8, 8, 8, 8, 8, 20],
        [2, 2, 2, 1, 2, 2, 8, 8, 8, 8, 8, 8, 20],
        [2, 2, 2, 2, 1, 2, 8, 8, 8, 8, 8, 8, 20],
        [2, 2, 2, 2, 2, 1, 8, 8, 8, 8, 8, 8, 20],
        [8, 8, 8, 8, 8, 8, 1, 2, 2, 2, 2, 2, 20], 
        [8, 8, 8, 8, 8, 8, 2, 1, 2, 2, 2, 2, 20],
        [8, 8, 8, 8, 8, 8, 2, 2, 1, 2, 2, 2, 20],
        [8, 8, 8, 8, 8, 8, 2, 2, 2, 1, 2, 2, 20],
        [8, 8, 8, 8, 8, 8, 2, 2, 2, 2, 1, 2, 20],
        [8, 8, 8, 8, 8, 8, 2, 2, 2, 2, 2, 1, 20],
        [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 1] ]
        D = torch.from_numpy(np.asarray(D).astype(np.float32)).to(pred.device)
        D -= D * torch.eye(13).to(pred.device)
        # return  ((D[targ] * (-(pred - 1.0).clamp (min = 1.e-7).log())).mean()) ** self.penalty
        # print("num: ", (D[targ] * pred).mean())
        return  (D[targ] * pred).mean()

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(x, y)
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, y)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def on_train_epoch_end(self):
        self.log('lr', self.optimizer_pointer.state_dict()['param_groups'][0]['lr'], sync_dist=True) 
        self.log("time", time.time() - self.start_time, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)  # , dropout_on=False
        loss = self.loss_fn(x, y)
        preds = torch.argmax(logits, dim=1)
        self.valid_acc(preds, y)
        self.log("valid_acc", self.valid_acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("valid_loss", loss, prog_bar=True, sync_dist=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)  #, dropout_on=False
        loss = self.loss_fn(x, y)
        preds = torch.argmax(logits, dim=1)
        self.test_acc(preds, y)
        self.log("test_acc", self.test_acc, prog_bar=True, sync_dist=True)  # let the hook determine the mode
        self.log("test_loss", loss, prog_bar=True, sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.l2)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_step, gamma=self.lr_gam)
        self.optimizer_pointer = optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch  # see predict.py, was running into an Attribute no Flatten error [resolved]
        logits = self(x)  # , dropout_on=False
        preds = torch.argmax(logits, dim=1)
        return preds
    
# Deep FCN experimentation
class DeepImprovedFCN(pl.LightningModule):    
    def __init__(self, input_size, num_classes, num_layers, hidden_layers, lr, lr_step, lr_gam, activation, penalty, dropout=None):
        super().__init__()
        self.save_hyperparameters()

        # PL attributes:
        self.train_acc = Accuracy(task='multiclass', num_classes=13)
        self.valid_acc = Accuracy(task='multiclass', num_classes=13)
        self.test_acc = Accuracy(task='multiclass', num_classes=13)

        # Parameters
        self.num_classes = num_classes
        self.num_layers = num_layers
        #self.hidden_layers = hidden_layers
        self.neurons_per_layer = hidden_layers
        self.lr = lr
        self.lr_step = lr_step
        self.lr_gam = lr_gam
        self.penalty = penalty
        self.dropout = dropout
        if dropout is not None: self.dropout_layer = nn.Dropout(dropout)
        self.activation = utils.get_activation(activation)

        # Other attributes
        self.start_time = time.time()
        self.optimizer_pointer = None

        self.num_blocks = int(self.num_layers / len(self.neurons_per_layer))
      
        # Defining model layers
        list_blocks = [nn.Flatten()]
        input = input_size
        for i in range(self.num_blocks):
            """IMPORTANt note on Architecture design: every block of hidden layers is designed to be identical
             since they are each using the same neurons_per_layer list of neuron dimensions.
             For example, if neurons_per_layer = [1024,512,256,128,64], every block of hidden layers will
             have these same neuron dimensions.
            """
            if i < self.num_blocks - 1:        
                first_size = self.neurons_per_layer[0]
                block_hidden, next_input = self.make_block(self.neurons_per_layer, input, first_size)
                # Append both to the ModuleLists
                list_blocks.append(block_hidden)
                input = next_input
            else:
                block_hidden, next_input = self.make_block(self.neurons_per_layer, input, self.num_classes)
                list_blocks.append(block_hidden)
        self.model = nn.Sequential(*list_blocks)

    # This method makes creates nn.Sequential objects
    def make_block(self, neurons_per_layer, input, output):
        normal_layers = [nn.Flatten()]
     
        # Loop to append linear layers
        inp = input
        # Dynamic hidden layers
        for num_neurons in neurons_per_layer:
            normal_layers.append(nn.Linear(inp, num_neurons))
            activator = self.activation
            normal_layers.append(activator)
            normal_layers.append(nn.Dropout(self.dropout))
            inp = num_neurons

        # Output layer to next block
        normal_layers.append(nn.Linear(neurons_per_layer[-1] , output))
        block_layers =  nn.Sequential(*normal_layers) 
        return block_layers, output

    def forward(self, x):
        x = self.model(x)
        return x

    # Give incorrect predictions a logarithmic divergence
    def misclassB (self, pred, targ):   # pred[nBatch, nClass], targ[nBatch], D[nClass, nClass]
        D = [ [1, 2, 2, 2, 2, 2, 8, 8, 8, 8, 8, 8, 20],
        [2, 1, 2, 2, 2, 2, 8, 8, 8, 8, 8, 8, 20],
        [2, 2, 1, 2, 2, 2, 8, 8, 8, 8, 8, 8, 20],
        [2, 2, 2, 1, 2, 2, 8, 8, 8, 8, 8, 8, 20],
        [2, 2, 2, 2, 1, 2, 8, 8, 8, 8, 8, 8, 20],
        [2, 2, 2, 2, 2, 1, 8, 8, 8, 8, 8, 8, 20],
        [8, 8, 8, 8, 8, 8, 1, 2, 2, 2, 2, 2, 20], 
        [8, 8, 8, 8, 8, 8, 2, 1, 2, 2, 2, 2, 20],
        [8, 8, 8, 8, 8, 8, 2, 2, 1, 2, 2, 2, 20],
        [8, 8, 8, 8, 8, 8, 2, 2, 2, 1, 2, 2, 20],
        [8, 8, 8, 8, 8, 8, 2, 2, 2, 2, 1, 2, 20],
        [8, 8, 8, 8, 8, 8, 2, 2, 2, 2, 2, 1, 20],
        [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 1] ]
        D = torch.from_numpy(np.asarray(D).astype(np.float32)).to(pred.device)
        D -= D * torch.eye(13).to(pred.device)
        return  ((D[targ] * (-(pred - 1.0).clamp (min = 1.e-7).log())).mean())  * self.penalty

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = (nn.functional.cross_entropy(self(x), y)) + self.misclassB(self(x), y)
        preds = torch.argmax(logits, dim=1)
        self.train_acc.update(preds, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def on_train_epoch_end(self):
        self.log("train_acc", self.train_acc.compute(), sync_dist=True)
        self.log('lr', self.optimizer_pointer.state_dict()['param_groups'][0]['lr'], sync_dist=True)  
        self.log("time", time.time() - self.start_time, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = (nn.functional.cross_entropy(self(x), y)) + self.misclassB(self(x), y)
        preds = torch.argmax(logits, dim=1)
        self.valid_acc.update(preds, y)
        self.log("valid_loss", loss, prog_bar=True, sync_dist=True)
        self.log("valid_acc", self.valid_acc.compute(), prog_bar=True, sync_dist=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = (nn.functional.cross_entropy(self(x), y)) + self.misclassB(self(x), y)
        preds = torch.argmax(logits, dim=1)
        self.test_acc.update(preds, y)
        self.log("test_loss", loss, prog_bar=True, sync_dist=True)
        self.log("test_acc", self.test_acc.compute(), prog_bar=True, sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=450, gamma=0.95)
        self.optimizer_pointer = optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch  # see predict.py, was running into an Attribute no Flatten error [resolved]
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        return preds
    
# LSTM 2023 with custom loss function inputs 
class ImprovedLSTM2023(pl.LightningModule):
    def __init__(self, indim, outdim, num_layers, neurons, lr, lr_step, lr_gam, dropout, activation, penalty, optimizer):
        super().__init__()
        self.save_hyperparameters()

        # PL attributes:
        self.train_acc = Accuracy(task='multiclass', num_classes=13)
        self.valid_acc = Accuracy(task='multiclass', num_classes=13)
        self.test_acc = Accuracy(task='multiclass', num_classes=13)

        self.indim = indim
        self.outdim = outdim
        self.num_layers = num_layers
        self.neurons = neurons
        self.lr = lr
        self.lr_step = lr_step
        self.lr_gam = lr_gam

        self.penalty = penalty
        self.input_layer = nn.Linear(self.indim, self.neurons)
        self.lstm = nn.LSTM(input_size=self.neurons, hidden_size=self.neurons, num_layers=4, batch_first=True)
        self.rnn_layers = self.get_layers()  # dense layers
        self.dropout = nn.Dropout(dropout)
        self.activation = utils.get_activation(activation)
        self.output_layer = nn.Linear(self.neurons, self.outdim)

        self.optimizer = utils.get_optimizer(optimizer)(self.parameters(), lr=lr)
        self.start_time = time.time()

    def forward(self, x, tgt, dropout_on=True):
        # x, tgt = batch
        x = x.unsqueeze(1)  # adds a sequence length of 1
        x = self.input_layer(x)
        
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(4, x.size(0), self.neurons).to(x.device).requires_grad_()
        # Initializing cell state for first input with zeros
        c0 = torch.zeros(4, x.size(0), self.neurons).to(x.device).requires_grad_()

        # Forward propagation by passing in the input, hidden state, and cell state into the model
        x, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        x = x[:, -1, :]

        # RNN layer
        for layer in self.rnn_layers: 
            if isinstance(layer, nn.Linear):
                x = layer(x)
            else:
                x, tgt = layer(x, tgt)
        
        x = self.activation(x) #activation function after fully connected layer
      
        # dropout
        if dropout_on:
            x = self.dropout(x)
        x = self.output_layer(x)
        return x

    def get_layers(self):
        layers = nn.ModuleList()
        layer = nn.Linear(self.neurons, self.neurons, bias=True)
        for _ in range(self.num_layers - 1):
            layers.append(layer)
        layers.append(layer)  
        return layers
    
    # Give incorrect predictions a logarithmic divergence
    def misclassB (self, pred, targ):   # pred[nBatch, nClass], targ[nBatch], D[nClass, nClass]
        D = [ [1, 2, 2, 2, 2, 2, 8, 8, 8, 8, 8, 8, 20],
        [2, 1, 2, 2, 2, 2, 8, 8, 8, 8, 8, 8, 20],
        [2, 2, 1, 2, 2, 2, 8, 8, 8, 8, 8, 8, 20],
        [2, 2, 2, 1, 2, 2, 8, 8, 8, 8, 8, 8, 20],
        [2, 2, 2, 2, 1, 2, 8, 8, 8, 8, 8, 8, 20],
        [2, 2, 2, 2, 2, 1, 8, 8, 8, 8, 8, 8, 20],
        [8, 8, 8, 8, 8, 8, 1, 2, 2, 2, 2, 2, 20], 
        [8, 8, 8, 8, 8, 8, 2, 1, 2, 2, 2, 2, 20],
        [8, 8, 8, 8, 8, 8, 2, 2, 1, 2, 2, 2, 20],
        [8, 8, 8, 8, 8, 8, 2, 2, 2, 1, 2, 2, 20],
        [8, 8, 8, 8, 8, 8, 2, 2, 2, 2, 1, 2, 20],
        [8, 8, 8, 8, 8, 8, 2, 2, 2, 2, 2, 1, 20],
        [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 1] ]
        D = torch.from_numpy(np.asarray(D).astype(np.float32)).to(pred.device)
        D -= D * torch.eye(13).to(pred.device)
        return  ((D[targ] * (-(pred - 1.0).clamp (min = 1.e-7).log())).mean())  * self.penalty

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x,y)
        loss = (nn.functional.cross_entropy(self(x,y), y)) + self.misclassB(self(x,y), y)
        preds = torch.argmax(logits, dim=1)
        self.train_acc.update(preds, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def on_train_epoch_end(self):
        self.log("train_acc", self.train_acc.compute(), sync_dist=True)
        self.log('lr', self.optimizer_pointer.state_dict()['param_groups'][0]['lr'], sync_dist=True) 
        self.log("time", time.time() - self.start_time, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x,y, dropout_on=False)
        loss = (nn.functional.cross_entropy(self(x,y), y)) + self.misclassB(self(x,y), y)
        preds = torch.argmax(logits, dim=1)
        self.valid_acc.update(preds, y)
        self.log("valid_loss", loss, prog_bar=True, sync_dist=True)
        self.log("valid_acc", self.valid_acc.compute(), prog_bar=True, sync_dist=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x,y, dropout_on=False)
        loss = (nn.functional.cross_entropy(self(x,y), y)) + self.misclassB(self(x,y), y)
        preds = torch.argmax(logits, dim=1)
        self.test_acc.update(preds, y)
        self.log("test_loss", loss, prog_bar=True, sync_dist=True)
        self.log("test_acc", self.test_acc.compute(), prog_bar=True, sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=450, gamma=0.95)
        self.optimizer_pointer = optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch  
        logits = self(x, torch.empty(x.size(dim=0)), dropout_on=False)  # SUPER HACKY, CHECK THIS
        
        preds = torch.argmax(logits, dim=1)
        return preds
    
# LSTM 2024 with custom loss function inputs
class ImprovedLSTM2024(pl.LightningModule):
    def __init__(self, 
                indim, outdim, 
                num_linears, neurons_per_hidden,
                input_neurons,  num_lstm_layers, hidden_state_size,
                lr, lr_step, lr_gam, 
                dropout, activation, 
                penalty, optimizer, l2=0.0, one_activation=False,
                custom_loss=True, bias=True):
        super().__init__()
        self.save_hyperparameters()

        # PL attributes
        self.train_acc = Accuracy(task='multiclass', num_classes=13)
        self.valid_acc = Accuracy(task='multiclass', num_classes=13)
        self.test_acc = Accuracy(task='multiclass', num_classes=13)
        self.validation_step_outputs = []

        # Architecture attributes
        self.indim = indim
        self.outdim = outdim
        self.num_linears = num_linears
        self.num_blocks = int(num_linears/len(neurons_per_hidden))
        # Neurons is the number of neurons of input linear layer to lstm layers
        self.input_neurons = input_neurons
        self.lr = lr
        self.lr_step = lr_step
        self.lr_gam = lr_gam
        self.l2 = l2
        self.penalty = penalty
        self.neurons_per_hidden = neurons_per_hidden
        self.hidden_state_size = hidden_state_size

        self.one_activation = one_activation
        # Bias is a boolean flag that will turn on bias for linear layers
        self.bias = bias
        self.activation = utils.get_activation(activation)
        self.dropout = nn.Dropout(dropout)
                  
        # Defining model layers
        self.input_layer = nn.Linear(self.indim, self.input_neurons)
        self.lstm = nn.LSTM(input_size=self.input_neurons, hidden_size=self.hidden_state_size, num_layers=num_lstm_layers, batch_first=True) #lstm layers
        self.fc_layers = self.get_layers(self.input_neurons)  # dense layers
        
        self.output_layer #= nn.Linear(self.neurons, self.outdim) I am leaving this uninit here and init in get_layers

        self.optimizer = utils.get_optimizer(optimizer)(self.parameters(), lr=lr, weight_decay=self.l2)
        self.start_time = time.time()

        # Custom loss is a boolean flag that will use custom loss in all steps
        self.custom_loss = custom_loss

    def forward(self, x):
        # x, tgt = batch
        x = x.unsqueeze(1)  # adds a sequence length of 1
        x = self.input_layer(x)
        
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(4, x.size(0), self.hidden_state_size).to(x.device).requires_grad_()
        # Initializing cell state for first input with zeros
        c0 = torch.zeros(4, x.size(0), self.hidden_state_size).to(x.device).requires_grad_()

        # Forward propagation by passing in the input, hidden state, and cell state into the model
        x, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        x = x[:, -1, :]

        x = self.fc_layers(x)
        x = self.output_layer(x)

        return x

    def get_layers(self, input):
        layers = [nn.Flatten()]
        
        # Input is the input size of the number of neurons in the (last) layer before this method is called
        indim = input
        for i in range(self.num_blocks):
            #IMPORTANT ARCHITECTURE DESIGN 
            # For example, neurons_per_layer = [256,128,64,32], num_linears = 8
            # Then the architecture of the hidden layers should be 
            # [Linear(256,128), Linear(128,64), Linear(64,32), 
            # Linear(32,256) <- where this last layer feeds into the next 'block' of hidden layers
            # Linear(256,128), Linear(128,64), Linear(64,32), Linear(32,num_classes) <-output layer ]
            # Iterate through neurons_per_layer and create connections of hidden layers with specified dimensions
            for num in self.neurons_per_hidden:
                # Outdim of current layer is next item in neurons_per_hidden
                outdim = num
                # Intializing with bias
                layer = nn.Linear(indim, outdim, bias=self.bias)
                 
                layers.append(layer)

                layers.append(self.dropout)
                if self.one_activation == False:
                    layers.append(self.activation)
                 
                # Reinitialze indim of next hidden layer as outdim of current layer
                indim = outdim

        if self.one_activation == True:
            layers.append(self.activation)
        self.output_layer = nn.Linear(outdim, self.outdim)
        hidden_layers = nn.Sequential(*layers)  
        return hidden_layers
    
    def loss_fn(self, x, y):
        # Return (nn.functional.cross_entropy(X, y)) + self.misclassB(X, y)
        cross_entropy = nn.functional.cross_entropy(self(x), y) 

        arr = torch.argmax(self(x), dim=1)
        encoded_arr = torch.zeros((arr.size(dim=0), 13), dtype=torch.int).to(x.device)
        encoded_arr[torch.arange(arr.size(dim=0)), arr] = 1
        pair_wise_x = encoded_arr

        #pair_wise_x = nn.Softmax(dim=1)(self(x))  

        pair_wise = (1 + self.loss_penalty(pair_wise_x, y)) ** self.penalty  
        return cross_entropy * pair_wise

    # Give incorrect predictions a logarithmic divergence
    def misclassB (self, pred, targ):   # pred[nBatch, nClass], targ[nBatch], D[nClass, nClass]
        D = [ [1, 2, 2, 2, 2, 2, 10, 10, 10, 10, 10, 10, 20],
        [2, 1, 2, 2, 2, 2, 10, 10, 10, 10, 10, 10, 20],
        [2, 2, 1, 2, 2, 2, 10, 10, 10, 10, 10, 10, 20],
        [2, 2, 2, 1, 2, 2, 10, 10, 10, 10, 10, 10, 20],
        [2, 2, 2, 2, 1, 2, 10, 10, 10, 10, 10, 10, 20],
        [2, 2, 2, 2, 2, 1, 10, 10, 10, 10, 10, 10, 20],
        [10, 10, 10, 10, 10, 10, 1, 2, 2, 2, 2, 2, 20], 
        [10, 10, 10, 10, 10, 10, 2, 1, 2, 2, 2, 2, 20],
        [10, 10, 10, 10, 10, 10, 2, 2, 1, 2, 2, 2, 20],
        [10, 10, 10, 10, 10, 10, 2, 2, 2, 1, 2, 2, 20],
        [10, 10, 10, 10, 10, 10, 2, 2, 2, 2, 1, 2, 20],
        [10, 10, 10, 10, 10, 10, 2, 2, 2, 2, 2, 1, 20],
        [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 1] ]
        D = torch.from_numpy(np.asarray(D).astype(np.float32)).to(pred.device)
        D -= D * torch.eye(13).to(pred.device)
        return  ((D[targ] * (-(pred - 1.0).clamp (min = 1.e-7).log())).mean()) * self.penalty
    
    # Experimenting with loss penalty
    def loss_penalty (self, pred, targ): 
        D = [ [1, 2, 2, 2, 2, 2, 10, 10, 10, 10, 10, 10, 20],
        [2, 1, 2, 2, 2, 2, 10, 10, 10, 10, 10, 10, 20],
        [2, 2, 1, 2, 2, 2, 10, 10, 10, 10, 10, 10, 20],
        [2, 2, 2, 1, 2, 2, 10, 10, 10, 10, 10, 10, 20],
        [2, 2, 2, 2, 1, 2, 10, 10, 10, 10, 10, 10, 20],
        [2, 2, 2, 2, 2, 1, 10, 10, 10, 10, 10, 10, 20],
        [10, 10, 10, 10, 10, 10, 1, 8, 2, 2, 2, 2, 20], 
        [10, 10, 10, 10, 10, 10, 10, 1, 8, 2, 2, 2, 20],
        [10, 10, 10, 10, 10, 8, 2, 8, 1, 8, 2, 2, 20],
        [10, 10, 10, 10, 10, 10, 2, 2, 8, 1, 8, 2, 20],
        [10, 10, 10, 10, 10, 10, 2, 2, 2, 8, 1, 8, 20],
        [10, 10, 10, 10, 10, 10, 2, 2, 2, 2, 8, 1, 20],
        [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 1] ]
        D = torch.from_numpy(np.asarray(D).astype(np.float32)).to(pred.device)
        D -= D * torch.eye(13).to(pred.device)
        # return  ((D[targ] * (-(pred - 1.0).clamp (min = 1.e-7).log())).mean()) ** self.penalty
        # print("num: ", (D[targ] * pred).mean())
        return  (D[targ] * pred).mean()

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        if self.custom_loss:
            loss = self.loss_fn(x, y)
        else:
            loss = (nn.functional.cross_entropy(self(x), y))
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, y)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.log('lr', self.optimizer_pointer.state_dict()['param_groups'][0]['lr'], sync_dist=True) 
        self.log("time", time.time() - self.start_time, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        if self.custom_loss:
            loss = self.loss_fn(x, y)
        else:
            loss = (nn.functional.cross_entropy(self(x), y))
        preds = torch.argmax(logits, dim=1)
        self.valid_acc(preds, y)
        self.log("valid_acc", self.valid_acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("valid_loss", loss, prog_bar=True, sync_dist=True)
        #self.log("valid_acc", self.valid_acc.compute(), prog_bar=True, sync_dist=True)
        #self.validation_step_outputs.append(loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        if self.custom_loss:
            loss = self.loss_fn(x, y)
        else:
            loss = (nn.functional.cross_entropy(self(x), y))
        preds = torch.argmax(logits, dim=1)
        self.test_acc(preds, y)
        self.log("test_acc", self.test_acc, prog_bar=True, sync_dist=True)  # let the hook determine the mode
        self.log("test_loss", loss, prog_bar=True, sync_dist=True)
        return loss
    

    def configure_optimizers(self):
        #optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        optimizer = self.optimizer
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=450, gamma=0.95)
        self.optimizer_pointer = optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch  # see predict.py, was running into an Attribute no Flatten error [resolved]
        # x, _ = batch  # see predict.py, was running into an Attribute no Flatten error [resolved]
        #logits = self(x, torch.empty(x.size(dim=0)), dropout_on=False)  # SUPER HACKY, CHECK THIS
        #x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        return preds


# Control test model by deleting LSTM layers
class LSTM2023Control(pl.LightningModule):

    def __init__(self, indim, outdim, num_layers, neurons, lr, lr_step, lr_gam, dropout, activation, optimizer, l2=0, log_test=False):
        super().__init__()
        self.save_hyperparameters()

        # PL attributes:
        self.train_acc = Accuracy(task='multiclass', num_classes=13)
        self.valid_acc = Accuracy(task='multiclass', num_classes=13)
        self.test_acc = Accuracy(task='multiclass', num_classes=13)

        self.indim = indim
        self.outdim = outdim
        self.num_layers = num_layers
        self.neurons = neurons
        self.lr = lr
        self.lr_step = lr_step
        self.lr_gam = lr_gam
        self.l2=l2

        self.input_layer = nn.Linear(self.indim, self.neurons)

        model_layers = []
        input_size = self.neurons
        for num in [512, 512, 376, 256]:  # hacky af
            layer = nn.Linear(input_size, num)
            model_layers.append(layer)
            model_layers.append(nn.ReLU())  
            print("\n\n\n\n\n\n\n\n\n\n\n\nFLAG NOTE: tanh changed!\n\n\n\n\n\n\n\n\n\n\n\n")
            input_size = num
        self.lstm = nn.Sequential(*model_layers)
        self.activation = utils.get_activation(activation)  
        self.rnn_layers = self.get_layers()  # dense layers
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(self.neurons, self.outdim)

        self.optimizer = utils.get_optimizer(optimizer)(self.parameters(), lr=lr, weight_decay=self.l2)
        self.start_time = time.time()
        self.log_test = log_test


    def forward(self, x, tgt, dropout_on=True):
        # x, tgt = batch
        x = x.unsqueeze(1)  # adds a sequence length of 1
        x = self.input_layer(x)
        
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(4, x.size(0), self.neurons).to(x.device).requires_grad_()
        # Initializing cell state for first input with zeros
        c0 = torch.zeros(4, x.size(0), self.neurons).to(x.device).requires_grad_()

        x = self.lstm(x)

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        x = x[:, -1, :]

        # RNN layer
        for layer in self.rnn_layers: 
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.ReLU):  
                x = layer(x)
            else:
                x, tgt = layer(x, tgt)
        
        x = self.activation(x) #activation function after fully connected layer

        # dropout
        if dropout_on:
            x = self.dropout(x)

        x = self.output_layer(x)
        return x

    def get_layers(self):
        layers = nn.ModuleList()
        layer = nn.Linear(self.neurons, self.neurons, bias=True)
        for _ in range(self.num_layers - 1):
            layers.append(layer)
        layers.append(layer)  
        return layers


    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x, y)
        loss = nn.functional.cross_entropy(self(x, y), y.view(-1))
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, y)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    

    def on_train_epoch_end(self):
        self.log('lr', self.optimizer.state_dict()['param_groups'][0]['lr'], sync_dist=True)  
        self.log("time", time.time() - self.start_time, sync_dist=True)


    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x, y, dropout_on=False)
        loss = nn.functional.cross_entropy(self(x, y), y.view(-1))
        preds = torch.argmax(logits, dim=1)
        self.valid_acc(preds, y)
        self.log("valid_acc", self.valid_acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("valid_loss", loss, prog_bar=True, sync_dist=True)
        return loss
    
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x, y, dropout_on=False)
        loss = nn.functional.cross_entropy(self(x, y), y.view(-1))
        preds = torch.argmax(logits, dim=1)
        self.test_acc(preds, y)
        self.log("test_acc", self.test_acc, prog_bar=True, sync_dist=True)  # let the hook determine the mode
        self.log("test_loss", loss, prog_bar=True, sync_dist=True)
        return loss
    

    def configure_optimizers(self):
        optimizer = self.optimizer
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_step, gamma=self.lr_gam)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
    

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch  
        logits = self(x, torch.empty(x.size(dim=0)), dropout_on=False)  
        preds = torch.argmax(logits, dim=1)
        return preds


# 1D CNN experimentation
class CNN1D(pl.LightningModule):    
    def __init__(self, input_size, num_classes, hidden_layers, activation, lr, lr_step, lr_gam, penalty, dropout=None, l2=0):
        super().__init__()
        self.save_hyperparameters()

        # PL attributes:
        self.train_acc = Accuracy(task='multiclass', num_classes=13)
        self.valid_acc = Accuracy(task='multiclass', num_classes=13)
        self.test_acc = Accuracy(task='multiclass', num_classes=13)

        # params
        self.input_size = input_size
        self.num_classes = num_classes
        
        self.hidden_layers = hidden_layers
        self.activation = utils.get_activation(activation)
        self.dropout = dropout
        if dropout is not None: self.dropout_layer = nn.Dropout(dropout)

        self.lr = lr
        self.lr_step = lr_step
        self.lr_gam = lr_gam
        self.l2 = l2

        self.penalty = penalty
        self.optimizer_pointer = None
        self.start_time = time.time()

        # CNN
        model_layers = [nn.Conv1d(1, 10, 1, stride=5), 
                        nn.Conv1d(10, 50, 1, stride=5), 
                        nn.Conv1d(50, 250, 1, stride=5),]
        self.cnn_model = nn.Sequential(*model_layers)
        input_size = 250

        # FCN
        model_layers = [nn.Flatten()]
        for num in self.hidden_layers:
            layer = nn.Linear(input_size, num)
            model_layers.append(layer)
            model_layers.append(self.activation)
            input_size = num
        self.fcn_model = nn.Sequential(*model_layers)
        self.output_layer = nn.Linear(self.hidden_layers[-1], num_classes)

    def to_img(self, x):
        img_x = torch.unsqueeze(x, 1)
        return img_x
    
    def forward(self, x, dropout_on=True):
        img_x = self.to_img(x)
        x = self.cnn_model(img_x)

        x = self.fcn_model(x)
        # dropout
        if self.dropout is not None and dropout_on:
            x = self.dropout_layer(x)       
        x = self.output_layer(x)
        return x
    
    def loss_fn(self, x, y):
        # return (nn.functional.cross_entropy(X, y)) + self.misclassB(X, y)
        cross_entropy = nn.functional.cross_entropy(self(x), y) 

        arr = torch.argmax(self(x), dim=1)
        encoded_arr = torch.zeros((arr.size(dim=0), 13), dtype=torch.int).to(x.device)
        encoded_arr[torch.arange(arr.size(dim=0)), arr] = 1
        pair_wise_x = encoded_arr

        #pair_wise_x = nn.Softmax(dim=1)(self(x)) 

        pair_wise = (1 + self.loss_penalty(pair_wise_x, y)) ** self.penalty  # put in actual predictions for custom loss
        return cross_entropy * pair_wise

    # give incorrect predictions a logarithmic divergence
    def misclassB (self, pred, targ):   # pred[nBatch, nClass], targ[nBatch], D[nClass, nClass]
        D = [ [1, 2, 2, 2, 2, 2, 8, 8, 8, 8, 8, 8, 20],
        [2, 1, 2, 2, 2, 2, 8, 8, 8, 8, 8, 8, 20],
        [2, 2, 1, 2, 2, 2, 8, 8, 8, 8, 8, 8, 20],
        [2, 2, 2, 1, 2, 2, 8, 8, 8, 8, 8, 8, 20],
        [2, 2, 2, 2, 1, 2, 8, 8, 8, 8, 8, 8, 20],
        [2, 2, 2, 2, 2, 1, 8, 8, 8, 8, 8, 8, 20],
        [8, 8, 8, 8, 8, 8, 1, 2, 2, 2, 2, 2, 20], 
        [8, 8, 8, 8, 8, 8, 2, 1, 2, 2, 2, 2, 20],
        [8, 8, 8, 8, 8, 8, 2, 2, 1, 2, 2, 2, 20],
        [8, 8, 8, 8, 8, 8, 2, 2, 2, 1, 2, 2, 20],
        [8, 8, 8, 8, 8, 8, 2, 2, 2, 2, 1, 2, 20],
        [8, 8, 8, 8, 8, 8, 2, 2, 2, 2, 2, 1, 20],
        [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 1] ]
        D = torch.from_numpy(np.asarray(D).astype(np.float32)).to(pred.device)
        D -= D * torch.eye(13).to(pred.device)
        return  ((D[targ] * (-(pred - 1.0).clamp (min = 1.e-7).log())).mean()) * self.penalty
    

    # experimenting with loss penalty
    def loss_penalty (self, pred, targ): 
        D = [ [1, 2, 2, 2, 2, 2, 8, 8, 8, 8, 8, 8, 20],
        [2, 1, 2, 2, 2, 2, 8, 8, 8, 8, 8, 8, 20],
        [2, 2, 1, 2, 2, 2, 8, 8, 8, 8, 8, 8, 20],
        [2, 2, 2, 1, 2, 2, 8, 8, 8, 8, 8, 8, 20],
        [2, 2, 2, 2, 1, 2, 8, 8, 8, 8, 8, 8, 20],
        [2, 2, 2, 2, 2, 1, 8, 8, 8, 8, 8, 8, 20],
        [8, 8, 8, 8, 8, 8, 1, 2, 2, 2, 2, 2, 20], 
        [8, 8, 8, 8, 8, 8, 2, 1, 2, 2, 2, 2, 20],
        [8, 8, 8, 8, 8, 8, 2, 2, 1, 2, 2, 2, 20],
        [8, 8, 8, 8, 8, 8, 2, 2, 2, 1, 2, 2, 20],
        [8, 8, 8, 8, 8, 8, 2, 2, 2, 2, 1, 2, 20],
        [8, 8, 8, 8, 8, 8, 2, 2, 2, 2, 2, 1, 20],
        [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 1] ]
        D = torch.from_numpy(np.asarray(D).astype(np.float32)).to(pred.device)
        D -= D * torch.eye(13).to(pred.device)
        # return  ((D[targ] * (-(pred - 1.0).clamp (min = 1.e-7).log())).mean()) ** self.penalty
        # print("num: ", (D[targ] * pred).mean())
        return  (D[targ] * pred).mean()

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(x, y)
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, y)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def on_train_epoch_end(self):
        self.log('lr', self.optimizer_pointer.state_dict()['param_groups'][0]['lr'], sync_dist=True) 
        self.log("time", time.time() - self.start_time, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x, dropout_on=False)
        loss = self.loss_fn(x, y)
        preds = torch.argmax(logits, dim=1)
        self.valid_acc(preds, y)
        self.log("valid_acc", self.valid_acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("valid_loss", loss, prog_bar=True, sync_dist=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x, dropout_on=False)
        loss = self.loss_fn(x, y)
        preds = torch.argmax(logits, dim=1)
        self.test_acc(preds, y)
        self.log("test_acc", self.test_acc, prog_bar=True, sync_dist=True)  
        self.log("test_loss", loss, prog_bar=True, sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_step, gamma=self.lr_gam)
        self.optimizer_pointer = optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch  # see predict.py, was running into an Attribute no Flatten error [resolved]
        logits = self(x, dropout_on=False)
        preds = torch.argmax(logits, dim=1)
        return preds

# ======================================================
# NEW 1D CNN Architecture 2025
class CNN1D_2025(pl.LightningModule):
    def __init__(self, input_size, num_classes, hidden_layers, activation, lr, lr_step, lr_gam, penalty, dropout=None, l2=0):
        super().__init__()
        self.save_hyperparameters()

        # PL attributes:
        self.train_acc = Accuracy(task='multiclass', num_classes=13)
        self.valid_acc = Accuracy(task='multiclass', num_classes=13)
        self.test_acc = Accuracy(task='multiclass', num_classes=13)

        # params
        self.input_size = input_size
        self.num_classes = num_classes

        self.hidden_layers = hidden_layers
        self.activation = utils.get_activation(activation)
        self.dropout = dropout
        if dropout is not None: self.dropout_layer = nn.Dropout(dropout)

        self.lr = lr
        self.lr_step = lr_step
        self.lr_gam = lr_gam
        self.l2 = l2

        self.penalty = penalty
        self.optimizer_pointer = None
        self.start_time = time.time()

          #CNN Model  - use padding to preserve input size
        model_layers = [
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),   # padding=1 to keep the size
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        ]

        # Adding MaxPool layer
        model_layers.append(nn.MaxPool1d(kernel_size=2, stride=2, padding=0))

        #CNN Block
        self.cnn_model = nn.Sequential(*model_layers)
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, self.input_size)  #(batch, channels, seq_len)
            cnn_output = self.cnn_model(dummy_input)
            input_size = cnn_output.view(1, -1).size(1)  #Flattened size

        #FCN layers
        model_layers_fc = [nn.Flatten()]
        for num in self.hidden_layers:
            layer = nn.Linear(input_size, num)
            model_layers_fc.append(layer)
            model_layers_fc.append(self.activation)  #Activation after  FCN layer
            input_size = num  #Update input size for next layer
        self.fcn_model = nn.Sequential(*model_layers_fc)

        # Output layer
        self.output_layer = nn.Linear(self.hidden_layers[-1], num_classes)

    def to_img(self, x):
        img_x = torch.unsqueeze(x, 1)
        return img_x

    def forward(self, x, dropout_on=True):
        img_x = self.to_img(x)
        x = self.cnn_model(img_x)

        x = self.fcn_model(x)
        # dropout
        if self.dropout is not None and dropout_on:
            x = self.dropout_layer(x)
        x = self.output_layer(x)
        return x

      def loss_fn(self, x, y):
        # return (nn.functional.cross_entropy(X, y)) + self.misclassB(X, y)
        cross_entropy = nn.functional.cross_entropy(self(x), y)

        arr = torch.argmax(self(x), dim=1)
        encoded_arr = torch.zeros((arr.size(dim=0), 13), dtype=torch.int).to(x.device)
        encoded_arr[torch.arange(arr.size(dim=0)), arr] = 1
        pair_wise_x = encoded_arr

        pair_wise = (1 + self.loss_penalty(pair_wise_x, y)) ** self.penalty  
        return cross_entropy * pair_wise


    # give incorrect predictions a logarithmic divergence
    def misclassB (self, pred, targ):   # pred[nBatch, nClass], targ[nBatch], D[nClass, nClass]
        D = [ [1, 2, 2, 2, 2, 2, 8, 8, 8, 8, 8, 8, 20],
        [2, 1, 2, 2, 2, 2, 8, 8, 8, 8, 8, 8, 20],
        [2, 2, 1, 2, 2, 2, 8, 8, 8, 8, 8, 8, 20],
        [2, 2, 2, 1, 2, 2, 8, 8, 8, 8, 8, 8, 20],
        [2, 2, 2, 2, 1, 2, 8, 8, 8, 8, 8, 8, 20],
        [2, 2, 2, 2, 2, 1, 8, 8, 8, 8, 8, 8, 20],
        [8, 8, 8, 8, 8, 8, 1, 2, 2, 2, 2, 2, 20],
        [8, 8, 8, 8, 8, 8, 2, 1, 2, 2, 2, 2, 20],
        [8, 8, 8, 8, 8, 8, 2, 2, 1, 2, 2, 2, 20],
        [8, 8, 8, 8, 8, 8, 2, 2, 2, 1, 2, 2, 20],
        [8, 8, 8, 8, 8, 8, 2, 2, 2, 2, 1, 2, 20],
        [8, 8, 8, 8, 8, 8, 2, 2, 2, 2, 2, 1, 20],
        [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 1] ]
        D = torch.from_numpy(np.asarray(D).astype(np.float32)).to(pred.device)
        D -= D * torch.eye(13).to(pred.device)
        return  ((D[targ] * (-(pred - 1.0).clamp (min = 1.e-7).log())).mean()) * self.penalty

    # experimenting with loss penalty
    def loss_penalty (self, pred, targ):
        D = [ [1, 2, 2, 2, 2, 2, 8, 8, 8, 8, 8, 8, 20],
        [2, 1, 2, 2, 2, 2, 8, 8, 8, 8, 8, 8, 20],
        [2, 2, 1, 2, 2, 2, 8, 8, 8, 8, 8, 8, 20],
        [2, 2, 2, 1, 2, 2, 8, 8, 8, 8, 8, 8, 20],
        [2, 2, 2, 2, 1, 2, 8, 8, 8, 8, 8, 8, 20],
        [2, 2, 2, 2, 2, 1, 8, 8, 8, 8, 8, 8, 20],
        [8, 8, 8, 8, 8, 8, 1, 2, 2, 2, 2, 2, 20],
        [8, 8, 8, 8, 8, 8, 2, 1, 2, 2, 2, 2, 20],
        [8, 8, 8, 8, 8, 8, 2, 2, 1, 2, 2, 2, 20],
        [8, 8, 8, 8, 8, 8, 2, 2, 2, 1, 2, 2, 20],
        [8, 8, 8, 8, 8, 8, 2, 2, 2, 2, 1, 2, 20],
        [8, 8, 8, 8, 8, 8, 2, 2, 2, 2, 2, 1, 20],
        [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 1] ]
        D = torch.from_numpy(np.asarray(D).astype(np.float32)).to(pred.device)
        D -= D * torch.eye(13).to(pred.device)
        # return  ((D[targ] * (-(pred - 1.0).clamp (min = 1.e-7).log())).mean()) ** self.penalty
        # print("num: ", (D[targ] * pred).mean())
        return  (D[targ] * pred).mean()

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(x, y)
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, y)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.log('lr', self.optimizer_pointer.state_dict()['param_groups'][0]['lr'], sync_dist=True)  # hacky, possibly damaging
        self.log("time", time.time() - self.start_time, sync_dist=True)


    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x, dropout_on=False)
        loss = self.loss_fn(x, y)
        preds = torch.argmax(logits, dim=1)
        self.valid_acc(preds, y)
        self.log("valid_acc", self.valid_acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("valid_loss", loss, prog_bar=True, sync_dist=True)
        return loss


    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x, dropout_on=False)
        loss = self.loss_fn(x, y)
        preds = torch.argmax(logits, dim=1)
        self.test_acc(preds, y)
        self.log("test_acc", self.test_acc, prog_bar=True, sync_dist=True)  # let the hook determine the mode
        self.log("test_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_step, gamma=self.lr_gam)
        self.optimizer_pointer = optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch  # see predict.py, was running into an Attribute no Flatten error [resolved]
        logits = self(x, dropout_on=False)
        preds = torch.argmax(logits, dim=1)
        return preds
