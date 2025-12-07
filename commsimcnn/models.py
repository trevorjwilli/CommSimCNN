import torch
from torch import nn

from commsimcnn.util import get_output_shape


class TinyVGG(nn.Module):
    def __init__(self,
                 input_shape: torch.Size,
                 output_shape: int,
                 in_channels: int = 1,
                 hidden_units: int = 10,
                 conv2d_kernel_size: int = 3,
                 maxpool_kernel_size: int = 2,
                 stride: int = 1,
                 padding: int = 0,
                 ):
        super().__init__()
        self.input_shape = input_shape
        self.conv2d_kernel_size = conv2d_kernel_size
        self.maxpool_kernel_size = maxpool_kernel_size
        self.height, self.width = self.calculate_output_shape()
        print(f"Output Height: {self.height} | Output Width: {self.width}")
        self.block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_units,
                kernel_size=conv2d_kernel_size,
                stride=stride,
                padding=padding
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=conv2d_kernel_size,
                stride=stride,
                padding=padding
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=maxpool_kernel_size)
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=conv2d_kernel_size,
                stride=stride,
                padding=padding
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=conv2d_kernel_size,
                stride=stride,
                padding=padding
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=maxpool_kernel_size)
        )
        self.classifer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*self.height*self.width,
                      out_features=output_shape)
        )

    def forward(self, x):
        """Execute the forward pass"""
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.classifer(x)
        return x
    
    def calculate_output_shape(self):
        """Get the final output shape from the CNN blocks"""
        out = self.input_shape

        # Output shape of block 1
        out = get_output_shape(out,
                               kernel_size=self.conv2d_kernel_size)
        out = get_output_shape(out,
                               kernel_size=self.conv2d_kernel_size)
        out = get_output_shape(out,
                               kernel_size=self.maxpool_kernel_size,
                               type='maxpool2d')
        
        # Output shape of block 2
        out = get_output_shape(out,
                               kernel_size=self.conv2d_kernel_size)
        out = get_output_shape(out,
                               kernel_size=self.conv2d_kernel_size)
        out = get_output_shape(out,
                               kernel_size=self.maxpool_kernel_size,
                               type='maxpool2d')
        
        return (out[0], out[1])