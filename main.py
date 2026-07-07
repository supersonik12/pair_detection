import sys
import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset, random_split
from EventDataset import EventDataset
from PairEventClassifier import PairEventClassifier
from PairEventCNN import PairEventCNN
from PairEvent2dCNN import PairEvent2dCNN

BATCH_SIZE = 64
# HIDDEN_DIM = ???

def MLP_run():    
    # Create EventDatasets from all data files provided and combine
    datasets = []
    for filename in sys.argv[1:]:
        dataset = EventDataset(filename)
        # function to adapt new dataset code (w more dimensions)
        dataset.flat()
        datasets.append(dataset)
    full_dataset = ConcatDataset(datasets)

    # this is torch.utils.data.random_split
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [0.8, 0.1, 0.1])

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print("Data loaded")

    # breaks if <1 thing in dataset but leaving it for now
    pair_count = 0
    total_count = 0
    for set in datasets:
        pair_count += set.get_counts()[0]
        total_count += set.get_counts()[1]
    pos_weight = (total_count - pair_count) / pair_count
    input_dim = len(train_dataset[0][0])
    print(f'Input dimensions: {input_dim}')

    model = PairEventClassifier(input_dim, pos_weight=pos_weight)

    print("Training model...")
    model.fit(trainloader=train_dataloader, validationloader=val_dataloader)

    print("Testing model...")
    accuracy = model.test(testloader=test_dataloader)
    print("Done")
    return accuracy

def CNN_run():    

    datasets = []
    for filename in sys.argv[1:]:
        dataset = EventDataset(filename)
        datasets.append(dataset)
    full_dataset = ConcatDataset(datasets)

    # this is torch.utils.data.random_split
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [0.8, 0.1, 0.1])

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print("Data loaded")

    # breaks if <1 thing in dataset but leaving it for now
    pair_count = 0
    total_count = 0
    for set in datasets:
        pair_count += set.get_counts()[0]
        total_count += set.get_counts()[1]
    pos_weight = (total_count - pair_count) / pair_count
    print(f'Proportion of negatives/positives = {pos_weight}')

    model = PairEvent2dCNN(pos_weight=1, lr=0.0001)

    # 'accelerator="auto"' automatically selects GPU/MPS if available, otherwise CPU
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        max_epochs=35)
    
    print("Training model...")
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    print("Testing model...")
    trainer.test(model, dataloaders=test_dataloader)

if __name__ == '__main__':
    if (len(sys.argv) < 2):
        print("Usage: py main.py <datafile1> [<datafile2> ...]")
        exit()

    # MLP_run()
    CNN_run()
