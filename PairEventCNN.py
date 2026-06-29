import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAccuracy, BinaryConfusionMatrix


class PairEventCNN(pl.LightningModule):
    def __init__(self, pos_weight=1.0, lr=0.001):
        super().__init__()
        self.lr = lr
        self.pos_weight = pos_weight

        # self.conv_net = nn.Sequential(
        #     nn.Conv2d(in_channels=2, out_channels=8, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(8),
        #     nn.ReLU(),
        #     nn.MaxPool2d((1,2)), # Warning: check kernel size
        #     nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2)
        # )

        # self.classifier = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Dropout(p=0.2),
        #     nn.Linear(1248, 1), # 624 = 156* 4, output layer flattened; 304 = max pool w a stride of (1,2)
        # )

        # trying with 1D Convolutional Network
        self.conv_net = nn.Sequential(
            nn.Conv1d(in_channels=8, out_channels=32, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2), # 156 -> 78
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2), # 78 -> 39
            
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
            # don't need to pool here bc we do it in the classifier
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(), # Output: [batch, 32]
            nn.Dropout(p=0.3),
            nn.Linear(128, 1)
        )

        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.pos_weight]))

        # Initialize tracking metrics for the test phase
        self.test_accuracy = BinaryAccuracy(threshold=0.40)
        self.test_confusion_matrix = BinaryConfusionMatrix(threshold=0.40)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 8, 156) # New shape: [batch, 8, 156]

        x = self.conv_net(x)
        x = self.classifier(x)
        return x.view(-1)

    # def _match_targets(self, labels, outputs):
    #     return labels.float().view_as(outputs)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
    
        loss = self.criterion(outputs, labels.float()) 
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

        if batch_idx == 0:
            print(f"\nProbabilities - Min: {predicted.min().item():.4f} | Max: {predicted.max().item():.4f} | Mean: {predicted.mean().item():.4f}")
     
        
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
