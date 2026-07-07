import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAccuracy, BinaryConfusionMatrix


class PairEvent2dCNN(pl.LightningModule):
    def __init__(self, pos_weight=1.0, lr=0.0001):
        super().__init__()
        self.lr = lr
        self.pos_weight = pos_weight

        self.wls_net = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=8, kernel_size=(1,11), stride=1, padding=5),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d((1,2)),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(1,5), stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((1,2)),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1,3), stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(80, 40),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(40, 1)
        )

        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.pos_weight]))

        # Initialize tracking metrics for the test phase
        self.test_accuracy = BinaryAccuracy()
        self.test_confusion_matrix = BinaryConfusionMatrix()


    def forward(self, x):
        batch_size = x.size(0)
        wls = self.wls_net(x[:, :, :, :150])
        edge = x[:, :, :, 150:].reshape(batch_size, 48, 1, 1)
        x = self.classifier(torch.cat((wls, edge), dim=1))
        return x.view(-1)


    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
    
        loss = self.criterion(outputs, labels) 
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss


    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
    
        loss = self.criterion(outputs, labels.float()) 
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss


    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        
        predicted = torch.sigmoid(outputs)
        # predicted = (outputs > 0.5).int().squeeze()

        # if batch_idx == 0:
        #     print(f"\nProbabilities - Min: {predicted.min().item():.4f} | Max: {predicted.max().item():.4f} | Mean: {predicted.mean().item():.4f}")
        
        # Update metrics for each batch
        self.test_accuracy.update(predicted, labels.int())
        self.test_confusion_matrix.update(predicted, labels.int())


    def on_test_epoch_end(self):
        # Compute aggregate scores across all evaluation batches
        final_acc = self.test_accuracy.compute()
        final_cm = self.test_confusion_matrix.compute()

        # Log total accuracy metrics
        print(f'\nTP: {final_cm[1, 1]}, FP: {final_cm[0, 1]}')
        print(f'FN: {final_cm[1, 0]}, TN: {final_cm[0, 0]}')
        print(f'Overall accuracy: {final_acc * 100:.3f}%\n')

        self.test_accuracy.reset()
        self.test_confusion_matrix.reset()


    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)


    def save(self, filepath):
        torch.save(self.state_dict(), filepath)
