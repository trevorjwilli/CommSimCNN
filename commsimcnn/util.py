import torch


def get_output_shape(x: torch.Size,
                     kernel_size: int | tuple,
                     stride: int | tuple = None ,
                     padding: int | tuple = 0,
                     dilation: int | tuple = 1,
                     type: str = 'Conv2d') -> torch.Size:
    """Function to calculate the output shape of a Conv2d layer

      Args:
          x (torch.Size): A torch.Size object indicating the shape of the incoming data
          kernel_size (int or tuple): Size of the convolving kernel
          stride (int or tuple, optional): Stride of the convolution. Defualt: 1
          padding (int or tuple, optional): Padding added to all four sides of the input. Default: 0
          dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
          type (str, optional): Which type of activation to use, one of "Conv2d" or "MaxPool2d"

      Details:
          This function takes the dimensions of a torch.tensor and calculates the output shape after passing through
          a Conv2D layer with the specified hyperparameters. For the equations used and more info on
          Conv2D, see the [Conv2d Documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)

      Returns:
          A 2L torch.Size object containing the dimensions of the height and width of the tensor
    """

    # Type assertions
    assert isinstance(x, torch.Size), "x must be an object of torch.Size"
    assert isinstance(kernel_size, int) | isinstance(kernel_size, tuple), "Expected int or tuple for kernel_size"
    assert isinstance(padding, int) | isinstance(padding, tuple), "Expected int or tuple for padding"
    assert isinstance(dilation, int) | isinstance(dilation, tuple), "Expected int or tuple for dilation"

    # Dimension length assertion
    assert len(x) >= 2, "Input object must have at least 2 dimensions"

    # Type assertion
    assert type.lower() in ['conv2d', 'maxpool2d'], "Expected one of 'Conv2d' or 'MaxPool2d' for type argument"

    if stride is None:
        if type.lower() == 'conv2d':
            stride = 1
        elif type.lower() == 'maxpool2d':
            stride = kernel_size

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    h_in = x[-2]
    w_in = x[-1]

    h_out_num = (h_in + 2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)
    h_out = int((h_out_num/stride[0]) + 1)
    
    w_out_num = (w_in + 2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)
    w_out = int((w_out_num/stride[1]) + 1)

    return torch.Size([h_out, w_out])