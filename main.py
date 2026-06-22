import sys
from torch.utils.data import DataLoader, ConcatDataset, random_split
from EventDataset import EventDataset
from PairEventClassifier import PairEventClassifier
from PairEventCNN import PairEventCNN

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
    train_dataset, valid_dataset, test_dataset = random_split(full_dataset, [0.8, 0.1, 0.1])

    # TODO: question: shuffle or not? why?
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print("Data loaded")

    # breaks if <1 thing in dataset but leaving it for now
    pair_count = 0
    total_count = 0
    for set in datasets:
        pair_count += set.get_counts()[0]
        total_count += set.get_counts()[1]
    pos_weight = pair_count / (total_count - pair_count)
    input_dim = len(train_dataset[0][0])
    print(f'Input dimensions: {input_dim}')

    model = PairEventClassifier(input_dim, pos_weight=pos_weight)

    print("Training model...")
    model.fit(trainloader=train_dataloader, validationloader=valid_dataloader)

    print("Testing model...")
    model.test(testloader=test_dataloader)
    print("Done")

if __name__ == '__main__':
    if (len(sys.argv) < 2):
        print("Usage: py main.py <datafile1> [<datafile2> ...]")
        exit()

    MLP_run()