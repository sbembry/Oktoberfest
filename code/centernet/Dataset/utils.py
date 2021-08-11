from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from .dataset import OktoberfestDataset

def load_train_data(path, random_state=1):
    """Loads csv, splits it by stack, and returns train and val as list"""
    with open(path) as f:
        data = f.readlines()
    del data[602] # bad way to handle an image with a bbox that is out of bounds
    stacks = list(set([d.split()[0].split('_')[0] for d in data]))
    train_stacks, val_stacks = train_test_split(list(stacks), random_state=random_state)
    train_stacks = set(train_stacks)
    val_stacks = set(val_stacks)

    train_data, val_data = [], []
    for d in data:
        stack = d.split()[0].split('_')[0]
        if stack in train_stacks:
            train_data.append(d)
        else:
            val_data.append(d)
    
    return train_data, val_data

def get_data_loader(data, path, augment=False, inference=False, shuffle=True, batch_size=16):
    ds = OktoberfestDataset(data, path=path, augment=augment, inference=inference)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
