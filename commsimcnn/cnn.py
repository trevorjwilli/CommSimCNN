import os
import torch
import torchmetrics
import torchmetrics.classification
from tqdm.auto import tqdm

import commsimcnn
from commsimcnn.models import *
from commsimcnn.simdataloader import SimDataLoader

class CNNModel():
    """
    Class to select, train, and evaluate a Convolutional Neural Network
    trained on species x site metacommunity matrices

    Attributes
    ----------

    Methods
    -------

    """
    def __init__(self,
                 train_data_path: str,
                 test_data_path: str,
                 loss_fn: torch.nn.Module,
                 optimizer: str = 'Adam',
                 learning_rate: float = 0.001,
                 train_transform = None,
                 test_transform = None,
                 batch_size: int = 32,
                 in_channels: int = 1,
                 hidden_units: int = 10,
                 conv2d_kernel_size: int | tuple = 3,
                 maxpool_kernel_size: int | tuple = 2,
                 conv2d_padding: int = 0,
                 maxpool_padding: int = 0,
                 conv2d_stride: int = 1):
        """
        Initialize class attributes
        
        Arguments
        ---------
        train_data_path: str
            Path to the directory containing the training data
        test_data_path: str
            Path to the directory containing the testing data
        train_transform:
            Torchvision.transforms to do on the training data
        test_transform:
            Torchvision.transoforms to do on the testing data
        batch_size: int
            Size of the batches to use in the torch.utils.data.DataLoaders
        in_channels: int
            Number of color channels in the image. For CommSim data, this will always
            be 1.
        hidden_units: int
            Number of neurons to use within the hidden layers.
        conv2d_kernel_size: int | tuple
            The size of the kernel for the convolutional layer.
        maxpool_kernel_size: int | tuple
            The size of the kernel for the maxpool layer. 
        stride: int
            The stride to use for the convolutional layer.
        padding: int
            The padding to use for the convolutional layer

        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.loss_fn = loss_fn
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.batch_size = batch_size
        self.num_workers = os.cpu_count() - 2 if os.cpu_count() > 2 else 1
        self.train_data = SimDataLoader(train_data_path, transform=train_transform)
        self.test_data = SimDataLoader(test_data_path, transform=test_transform)
        self.classes = self.train_data.classes
        self.class_to_idx = self.train_data.class_to_idx
        self.train_dataloader, self.test_dataloader = self.create_dataloaders()
        self.in_channels = in_channels
        self.hidden_units = hidden_units
        self.conv2d_kernel_size = conv2d_kernel_size
        self.maxpool_kernel_size = maxpool_kernel_size
        self.conv2d_stride = conv2d_stride
        self.conv2d_padding = conv2d_padding
        self.maxpool_padding = maxpool_padding
        self.model = self.create_model_instance()
        self.optimizer = self.set_optimizer(optimizer, learning_rate)
        self.train_loss = []
        self.train_acc = []
        self.train_f1 = []
        self.train_auroc = []
        self.test_loss = []
        self.test_acc = []
        self.test_f1 = []
        self.test_auroc = []
        

    def create_dataloaders(self):
        """Create the training and testing dataloaders"""

        train_dataloader = torch.utils.data.DataLoader(dataset=self.train_data,
                                                       batch_size=self.batch_size,
                                                       num_workers=self.num_workers,
                                                       shuffle=True)
        
        test_dataloader = torch.utils.data.DataLoader(dataset=self.test_data,
                                                      batch_size=self.batch_size,
                                                      num_workers=self.num_workers,
                                                      shuffle=False)
        
        return train_dataloader, test_dataloader
    

    def create_model_instance(self):
        """Creates the model instance"""

        input_shape = next(iter(self.train_dataloader))[0].shape
        output_shape = len(self.train_data.classes)

        model = TinyVGG(input_shape=input_shape,
                        output_shape=output_shape,
                        in_channels=self.in_channels,
                        hidden_units=self.hidden_units,
                        conv2d_kernel_size=self.conv2d_kernel_size,
                        maxpool_kernel_size=self.maxpool_kernel_size,
                        conv2d_stride=self.conv2d_stride,
                        conv2d_padding=self.conv2d_padding,
                        maxpool_padding=self.maxpool_padding

        ).to(self.device)

        return model
    

    def set_optimizer(self, optimizer, learning_rate):
        """Sets up the optimizer"""
        optimizer = optimizer.lower()

        assert optimizer in ('sgd', 'adam'), "Optimizer must be one of 'SGD', 'Adam'"

        if optimizer == 'sgd':
            optim = torch.optim.SGD(params=self.model.parameters(),
                                          lr=learning_rate)
        elif optimizer == 'adam':
            optim = torch.optim.Adam(params=self.model.parameters(),
                                          lr=learning_rate)
        return optim

    
    def train_step(self):
        """Train the model for a single epoch"""
        # Put model in train mode
        self.model.train()

        # Setup train loss 
        train_loss = 0

        # Setup accuracy metric
        acc_metric = torchmetrics.classification.Accuracy(task='multiclass',
                                                      num_classes = len(self.classes)).to(self.device)
        f1_metric = torchmetrics.classification.F1Score(task='multiclass',
                                                        num_classes=len(self.classes),
                                                        average='weighted').to(self.device)
        auroc_metric = torchmetrics.classification.AUROC(task="multiclass",
                                                         num_classes=len(self.classes)).to(self.device)
        for X, y in self.train_dataloader:
            X, y = X.to(self.device), y.to(self.device)

            y_logits = self.model(X)
            y_pred = torch.softmax(y_logits, dim=1)
            y_pred_class = torch.argmax(y_pred, dim=1)

            loss = self.loss_fn(y_logits, y)
            train_loss += loss.item()

            acc = acc_metric(y_pred_class, y)
            f1 = f1_metric(y_pred_class, y)
            auroc = auroc_metric(y_pred, y)

            # Optimzer zero grad
            self.optimizer.zero_grad()

            # Back propagation
            loss.backward()
            
            # optimizer step
            self.optimizer.step()

        train_loss = train_loss / len(self.train_dataloader)
        acc = acc_metric.compute()
        f1 = f1_metric.compute()
        auroc = auroc_metric.compute()

        self.train_loss.append(train_loss)
        self.train_acc.append(acc.to('cpu'))
        self.train_f1.append(f1.to('cpu'))
        self.train_auroc.append(auroc.to('cpu'))
    

    def test_step(self):
        """Run a single step of the test loop"""

        # put model in eval mode
        self.model.eval()

        # Set up metrics
        test_loss = 0
        acc_metric = torchmetrics.classification.Accuracy(task='multiclass',
                                                      num_classes = len(self.classes)).to(self.device)
        f1_metric = torchmetrics.classification.F1Score(task='multiclass',
                                                        num_classes=len(self.classes),
                                                        average='weighted').to(self.device)
        auroc_metric = torchmetrics.classification.AUROC(task="multiclass",
                                                         num_classes=len(self.classes)).to(self.device)
        with torch.inference_mode():
            for X, y in self.test_dataloader:
                X, y = X.to(self.device), y.to(self.device)

                y_logits = self.model(X)
                y_pred = torch.softmax(y_logits, dim=1)
                y_pred_class = torch.argmax(y_pred, dim=1)

                loss = self.loss_fn(y_logits, y)
                test_loss += loss.item()
                acc = acc_metric(y_pred_class, y)
                f1 = f1_metric(y_pred_class, y)
                auroc = auroc_metric(y_pred, y)               

        test_loss = test_loss / len(self.test_dataloader)
        acc = acc_metric.compute()
        f1 = f1_metric.compute()
        auroc = auroc_metric.compute()

        self.test_loss.append(test_loss)
        self.test_acc.append(acc.to('cpu'))
        self.test_f1.append(f1.to('cpu'))
        self.test_auroc.append(auroc.to('cpu'))

    def train(self, epochs: int):
        """Train the model for specified number of epochs"""
        for epoch in tqdm(range(epochs)):
            self.train_step()
            self.test_step()

            print(f"\nEpoch: {epoch}")
            print(f"Train Loss: {self.train_loss[epoch]:.4f} | Test Loss: {self.test_loss[epoch]:.4f}")
            print(f"Train Accuracy: {self.train_acc[epoch]:.4f} | Test Accuracy: {self.test_acc[epoch]:.4f}")
            print(f"Train F1: {self.train_f1[epoch]:.4f} | Test F1: {self.test_f1[epoch]:.4f}")
            print(f"Train AUROC: {self.train_auroc[epoch]:.4f} | Test Loss: {self.test_auroc[epoch]:.4f}\n")

    