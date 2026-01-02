import os
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class SimDataLoader(Dataset):
    """
    Class to load in simulated communities (saved as csv.gz) into a torch Dataset

    Attributes
    ----------
    root_dir : str
        The target directory containing the data 
    paths : list
        List containing the paths to each simulated dataset
    transform : torchvision.transforms
        The torchvision transforms to use on the data
    classes : list
        The list of label classes (targets)
    class_to_idx : dict
        Dictionary where class labels are keys and the index of the label is the value

    Methods
    -------
    get_classes():
        Extracts class labels from directory file structure and returns a list
        of class labels and a dictionary of class labels to indices
    load_sim(index: int):
        Loads a single simulation and returns it as a numpy array. Input csv.gz's
        are expected to contain counts of species in communities. This method 
        transforms these counts to community frequencies by dividing by community
        sizes
    __len__():
        Method to get the number of simulations. This method overwrites
        Dataset.__len__
    __getitem__(index: int):
        Method to get and load a single simulation. This method overwrites
        Dataset.__getitem__
    """

    def __init__(self,
                 target_dir: str,
                 transform: transforms = None):
        """
        Constructs necessary attributes for SimDataLoader object

        Arguments
        ---------
        target_dir : str
            The root directory containing the simulated data
        transform : torchvision.transforms
            The transforms to use on the data once imported
        """

        self.root_dir = target_dir
        # Get list of all simulations
        self.paths = list(Path(target_dir).glob("*/*.csv.gz"))
        # Setup transforms
        self.transform = transform
        # Create classes and class_to_idx attributes
        self.classes, self.class_to_idx = self.get_classes()

    def get_classes(self):
        """Method to extract class names and create class dictionary"""

        # Get class names and store as list
        classes = sorted([entry.name for entry in os.scandir(self.root_dir) if entry.is_dir()])

        if not classes:
            raise FileNotFoundError(f"Couldn't find any classes in {self.root_dir}... please check file structure.")
        # Create dictionary for class name to index
        class_to_idx = {class_name: i for i, class_name in enumerate(classes)}
        return classes, class_to_idx
    
    def load_sim(self, index: int) -> np.array:
        """Method to load a single simulation and return it as a numpy array"""
        sim_path = self.paths[index]
        # read in data
        data = pd.read_csv(sim_path)
        # compute row sums (i.e. community size)
        row_sums = data.sum(axis=1)
        # convert counts to species frequencies by dividing by community size
        data = data.div(row_sums, axis=0)
        # convert to array then to tensor
        data = data.to_numpy(dtype=float)
        data = torch.Tensor(data).unsqueeze(dim=0)
        return data
    
    def __len__(self) -> int:
        """Returns the total number of samples"""
        return len(self.paths)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """Returns one simulation, data and label (X, y)"""
        sim = self.load_sim(index)
        class_name = self.paths[index].parent.name # expects path in format: data_folder/class_name/image.jpg
        class_idx = self.class_to_idx[class_name]

        # Transform if necessary
        if self.transform:
            return self.transform(sim), class_idx # return data, label (X, y)
        else:
            return sim, class_idx # return untransformed image and label
    

