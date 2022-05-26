from torch import Tensor
from torch.utils.data.dataset import Dataset
from torch.utils.data import random_split, DataLoader
import pandas as pd

class Iris(Dataset):
    """
    Datasets are apparently dealt with as a class?
    __init__(): -> Do all of your preproccessing here.
        Seems like most people use Pandas/PySpark here
        You will need numpy representations/Tensor representations to get things to work correctly
    """
    def __init__(self):
        df = pd.read_csv("IRIS.csv")

        # Option 1: use df.pop
        self.y = df.pop('species')
        self.x = df
        # Option 2: Use df.drop(columns=["species"])
        # x_df = df.drop(columns=["species"])
        # y_df = df["species"]

        # Tensors must be homogeneous, numerical data type
        # Pandas/PySpark likely one of the tools needed before feeding
        # into a model.
        mapper = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
        self.y = self.y.replace(mapper)

        self.x = Tensor(self.x.to_numpy()).float()
        self.y = self.y.to_numpy()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i:int):
        return self.x[i], self.y[i]

class IrisDataLoader():
    """
    The dataloader class shuffles and manages batches.
    For larger projects, it sounds like the PyTorch-specific commands are used often
    Some outside process will do the batching and push to a file. Look at distributed computing libraries
    such as ray/dask.
    """
    def __init__(self, batch_size, training_frac=0.8):
        data = Iris()
        train_len, test_len = int(len(data) * training_frac), len(data) - int(len(data) * training_frac)
        train_dataset, test_dataset = random_split(data, [train_len, test_len])
        
        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        self.test_dataloader = DataLoader(test_dataset, batch_size=batch_size)