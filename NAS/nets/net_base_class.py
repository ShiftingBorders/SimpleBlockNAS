import lightning as L
import torch
import torch.nn as nn
from torchmetrics import Accuracy
import logging
import warnings
logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", ".*does not have many workers.*")
class ModelTrainer:

    def __init__(self, val_every_n_epoch = 2):
        self.logger = False
        self.validation_freq = val_every_n_epoch

    def train_net(self, net, accelerator, precision, train_loader, test_loader, epoch_num):
        accelerator = accelerator
        if accelerator == 'cpu':
            trainer = L.Trainer(
                max_epochs=epoch_num,
                accelerator=accelerator,
                precision=precision,
                logger=self.logger,
                enable_checkpointing=False,
                enable_model_summary=False,
                enable_progress_bar=False,
                check_val_every_n_epoch=self.validation_freq
            )
        if accelerator == 'gpu':
            trainer = L.Trainer(
                max_epochs=epoch_num,
                accelerator=accelerator,
                precision=precision,
                logger=self.logger,
                enable_checkpointing=False,
                enable_model_summary=False,
                enable_progress_bar=False,
                check_val_every_n_epoch= self.validation_freq
            )
        trainer.fit(net, train_loader, test_loader)
        return self.model_data(net)

    def extract_weights(self,net_modules):
        weights = []
        biases = []
        for layer in net_modules.children():
            weights_layer = None
            biases_layer = None
            if isinstance(layer, (nn.BatchNorm2d, nn.MaxPool2d,  nn.Dropout)):
                continue
            if hasattr(layer, 'weight'):
                if layer.weight is not None:
                    weights_layer = layer.weight.detach().cpu()
            else:
                pass

            if hasattr(layer, 'bias'):
                if layer.bias is not None:
                    biases_layer = layer.bias.detach().cpu()
            else:
                pass

            if weights_layer is None and biases_layer is None:
                continue
            weights.append(weights_layer)
            biases.append(biases_layer)

        return weights, biases

    def model_data(self, net):
        loss_data = net.loss_val_hist
        acc_data = net.acc_val_hist
        weights, biases = self.extract_weights(net.net)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        return {'lowest_loss: ': min(loss_data), 'highest_acc': max(acc_data), 'loss': loss_data, 'accuracy': acc_data, 'weights': weights, 'biases': biases}

class CNNClassifier(L.LightningModule):

    def __init__(self, *args):
        super().__init__()

        payload = args[0]
        self.loss_fn = nn.CrossEntropyLoss()
        self.acc_fn = Accuracy(task='multiclass', num_classes=payload['num_classes'])
        self.lr = payload['lr']
        self.loss_train_hist = []
        self.loss_val_hist = []
        self.acc_train_hist = []
        self.acc_val_hist = []
        self.lr = 1e-3
        self.batch_num = 0
        self.num_samples = 0

        self.running_loss_train = 0.0
        self.running_loss_val = 0.0

        self.reset_net_arch(payload['blocks'])

    def reset_net_arch(self, blocks):
        #self.net = None
        net = []
        for layers in blocks:
            layers_ = layers.get_layer()
            for g in layers_:
                net.append(g)
        self.net = nn.ModuleList(net)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(0.9, 0.999))
        return optimizer

    def forward_pass(self, x):
        previous_out = x
        for i in range(len(self.net)):
            part_output = self.net[i](previous_out)
            previous_out = part_output
        return previous_out

    def on_train_epoch_start(self):
        self.running_loss_train = 0.0
        self.batch_num = 0
        self.acc_fn.reset()

    def training_step(self, batch, batch_idx):
        data, targets = batch
        output = self.forward_pass(data)
        targets = targets.long()
        loss = self.loss_fn(output, targets)
        self.acc_fn.update(output, targets)
        self.running_loss_train += loss.detach()
        self.batch_num += 1
        return loss

    def on_train_epoch_end(self):
        final_loss = self.running_loss_train.item() / self.batch_num
        final_acc = self.acc_fn.compute().item()
        self.loss_train_hist.append(final_loss)
        self.acc_train_hist.append(final_acc)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def on_validation_epoch_start(self):
        self.running_loss_val = 0
        self.batch_num = 0
        self.acc_fn.reset()

    def validation_step(self, batch, batch_idx):
        data, targets = batch
        output = self.forward_pass(data)
        targets = targets.long()
        loss = self.loss_fn(output, targets)
        self.acc_fn.update(output, targets)
        self.running_loss_val += loss.detach()
        self.batch_num += 1
        return loss

    def on_validation_epoch_end(self):
        final_loss = self.running_loss_val.item() / self.batch_num
        final_acc = self.acc_fn.compute().item()
        self.loss_val_hist.append(final_loss)
        self.acc_val_hist.append(final_acc)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
