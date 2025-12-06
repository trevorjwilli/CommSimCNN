import pytest
import torch

from commsimcnn.util import get_output_shape

@pytest.fixture
def outputdata():
    out = get_output_shape(torch.Size([32, 3, 28, 28]),
                           kernel_size=3)
    return out


class TestConv2dFunction:
    
    def test_output_correct_type(self, outputdata):
        assert isinstance(outputdata, torch.Size)

    def test_output_correct_len(self, outputdata):
        assert len(outputdata) == 2

    def test_output_correct(self, outputdata):
        assert outputdata == torch.Size([26, 26])

    def test_output_maxpool2(self):
        out = get_output_shape(torch.Size([32, 3, 28, 28]),
                               kernel_size=2,
                               type='MaxPool2d')
        assert out == torch.Size([14, 14])

    def test_assertion_type(self):
        with pytest.raises(AssertionError, match=r"Expected int or tuple .*"):
            get_output_shape(torch.Size([1, 28, 28]),
                                     kernel_size=2,
                                     padding='a')
    
    def test_assertion_length(self):
        with pytest.raises(AssertionError, match=r".* at least 2 .*"):
            get_output_shape(torch.Size([1]), kernel_size=3)

    def test_assertion_type_arg(self):
        with pytest.raises(AssertionError, match=r".* 'MaxPool2d' .*"):
            get_output_shape(torch.Size([32, 32]), kernel_size=3,
                             type='Hello')
            