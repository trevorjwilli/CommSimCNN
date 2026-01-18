import torch
from torch import nn

from commsimcnn.util import get_output_shape


class TinyVGG(nn.Module):
    """
    Class for a CNN using the TinyVGG architecture

    Attributes
    ----------
    input_shape : torch.Size
        The shape (dimensions) of the input data to be trained on. The expected
        order of the dimensions is color_channels x height x width
    output_shape : int
        The number of classes for the model to predict on
    in_channels : int (default=1)
        The number of color_channels for the input data
    hidden_units : int (default=10)
        The number of neurons in the hidden layers of the model
    conv2d_kernel_size : int or tuple (default=3)
        The kernel size to use for the Conv2D layers. All four layers (two in each block)
        use the same kernel size
    maxpool_kernel_size : int or tuple (default=2)
        The kernel size to use for the MaxPool2D layers. Both max pool layers use the 
        same kernel size
    stride : int
        The stride to use in the Conv2D layers. For maxpool layers the stride is 
        equal to the kernel size
    padding : int

    Methods
    -------
    forward(x):
        Conducts a forward pass of the model using the input data
    calculate_output_shape():
        Calculates the final shape of the convulational blocks for resisizing in the classifier
    """
    def __init__(self,
                 input_shape: torch.Size,
                 output_shape: int,
                 in_channels: int = 1,
                 hidden_units: int = 10,
                 conv2d_kernel_size: int | tuple = 3,
                 maxpool_kernel_size: int | tuple = 2,
                 conv2d_padding: int | tuple = 0,
                 maxpool_padding: int | tuple = 0,
                 conv2d_stride: int | tuple = 1,
                 ):
        super().__init__()
        self.input_shape = input_shape
        self.conv2d_kernel_size = conv2d_kernel_size
        self.maxpool_kernel_size = maxpool_kernel_size
        self.conv2d_padding = conv2d_padding
        self.maxpool_padding = maxpool_padding
        self.conv2d_stride = conv2d_stride
        self.maxpool_stride = maxpool_kernel_size
        self.height, self.width = self.calculate_output_shape()
        # print(f"Output Height: {self.height} | Output Width: {self.width}")
        self.block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_units,
                kernel_size=self.conv2d_kernel_size,
                stride=self.conv2d_stride,
                padding=self.conv2d_padding
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=self.conv2d_kernel_size,
                stride=self.conv2d_stride,
                padding=self.conv2d_padding
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=self.maxpool_kernel_size,
                padding=self.maxpool_padding)
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=self.conv2d_kernel_size,
                stride=self.conv2d_stride,
                padding=self.conv2d_padding
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=self.conv2d_kernel_size,
                stride=self.conv2d_stride,
                padding=self.conv2d_padding
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=self.maxpool_kernel_size,
                padding=self.maxpool_padding)
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
                               kernel_size=self.conv2d_kernel_size,
                               padding=self.conv2d_padding,
                               stride=self.conv2d_stride)

        out = get_output_shape(out,
                               kernel_size=self.conv2d_kernel_size,
                               padding=self.conv2d_padding,
                               stride=self.conv2d_stride)
        
        out = get_output_shape(out,
                               kernel_size=self.maxpool_kernel_size,
                               padding=self.maxpool_padding,
                               stride=self.maxpool_stride,
                               type='maxpool2d')
        
        # Output shape of block 2
        out = get_output_shape(out,
                               kernel_size=self.conv2d_kernel_size,
                               padding=self.conv2d_padding,
                               stride=self.conv2d_stride)
        out = get_output_shape(out,
                               kernel_size=self.conv2d_kernel_size,
                               padding=self.conv2d_padding,
                               stride=self.conv2d_stride)
        out = get_output_shape(out,
                               kernel_size=self.maxpool_kernel_size,
                               padding=self.maxpool_padding,
                               stride=self.maxpool_stride,
                               type='maxpool2d')
        
        return (out[0], out[1])