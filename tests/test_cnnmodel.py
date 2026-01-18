import pytest
import torch

from commsimcnn.cnn import CNNModel

@pytest.fixture
def cnn_instance():
    return CNNModel(
        train_data_path='tests/test_data/train',
        test_data_path='tests/test_data/test',
        loss_fn=torch.nn.CrossEntropyLoss(),
        conv2d_kernel_size=2,
        maxpool_kernel_size=2,
    )
    
def test_train_step(cnn_instance):
    original_acc_len = len(cnn_instance.train_acc)
    cnn_instance.train_step()
    # Ensure the train step writes out values to lists
    assert len(cnn_instance.train_acc) > original_acc_len
    # Ensure the train step writes the correct data type
    assert isinstance(cnn_instance.train_acc[0], torch.Tensor)

def test_test_step(cnn_instance):
    original_auroc_len = len(cnn_instance.train_acc)
    cnn_instance.train_step()
    assert len(cnn_instance.train_auroc) > original_auroc_len
    assert isinstance(cnn_instance.train_acc[0], torch.Tensor)

            
            
            
            