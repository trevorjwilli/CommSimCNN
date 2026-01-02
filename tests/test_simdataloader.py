import pytest
import torch

from commsimcnn.simdataloader import SimDataLoader

@pytest.fixture
def load_train_data():
    train_data = SimDataLoader('tests/test_data/train')
    return train_data

@pytest.fixture
def load_test_data():
    test_data = SimDataLoader('tests/test_data/test')
    return test_data

class TestSimDataLoader:

    def test_len_method(self,
                        load_train_data,
                        load_test_data):
        assert len(load_train_data) == 160
        assert len(load_test_data) == 40
    
    def test_getitem_method(self,
                            load_train_data,
                            load_test_data):
        train_data = load_train_data
        train_item = train_data[0]
        assert isinstance(train_item, tuple)
        assert len(train_item) == 2
        assert isinstance(train_item[0], torch.Tensor)
        assert isinstance(train_item[1], int)
        assert train_item[0].shape == torch.Size((1, 15, 10))

        test_data = load_test_data
        test_item = test_data[0]
        assert isinstance(test_item, tuple)
        assert len(test_item) == 2
        assert isinstance(test_item[0], torch.Tensor)
        assert isinstance(test_item[1], int)        
        assert test_item[0].shape == torch.Size((1, 15, 10))

    def test_classes_attribute(self,
                               load_train_data,
                               load_test_data):
        train_data = load_train_data
        assert train_data.classes
        assert len(train_data.classes) == 2
        assert 'NTNDL' in train_data.classes
        assert 'SSNDL' in train_data.classes

        test_data = load_test_data
        assert test_data.classes
        assert len(test_data.classes) == 2
        assert 'NTNDL' in test_data.classes
        assert 'SSNDL' in test_data.classes

    def test_class_to_idx_attribute(self,
                                    load_train_data,
                                    load_test_data):
        train_data = load_train_data
        assert train_data.class_to_idx
        assert isinstance(train_data.class_to_idx, dict)
        assert train_data.class_to_idx['SSNDL'] == 1
        assert train_data.class_to_idx['NTNDL'] == 0

        test_data = load_test_data
        assert test_data.class_to_idx
        assert isinstance(test_data.class_to_idx, dict)
        assert test_data.class_to_idx['SSNDL'] == 1
        assert test_data.class_to_idx['NTNDL'] == 0