from torch.utils.data import DataLoader

from iterativenn.utils.DataModules import LMDataModuleGPT2
from iterativenn.utils.DataModules import MNISTRepeatedSequenceDataModule


def test_MNISTRepeatedSequenceDataModule():
    data_module = MNISTRepeatedSequenceDataModule(batch_size=17, train_size=51, min_copies=2, max_copies=4)
    data_module.prepare_data()
    data_module.setup('fit')
    assert len(data_module.train_dataloader()) == 3, "MNISTRepeatedSequenceDataModule should have 3 batches"
    batch = next(iter(data_module.train_dataloader()))
    assert len(batch) == 17, "MNISTRepeatedSequenceDataModule should have batch size 17"
    assert len(batch[0]['x']) >= 2 and len(
        batch[0]['x']) <= 4, "MNISTRepeatedSequenceDataModule x should be a sequence of length 2 to 4"
    assert len(batch[0]['y']) >= 2 and len(
        batch[0]['y']) <= 4, "MNISTRepeatedSequenceDataModule y should be a sequence of length 2 to 4"
    assert batch[0]['x'][0].shape[1] == 28, "MNISTRepeatedSequenceDataModule should have 28x28 images"
    assert batch[0]['x'][0].shape[2] == 28, "MNISTRepeatedSequenceDataModule should have 28x28 images"



def test_LMDataModuleGPT2_train_dataloader():
    data_module = LMDataModuleGPT2(
        model_name_or_path="gpt2",
        pad_to_max_length=True,
        preprocessing_num_workers=4,
        overwrite_cache=False,
        max_seq_length=128,
        mlm_probability=0.15,
        train_batch_size=16,
        val_batch_size=16,
        dataloader_num_workers=1,
    )
    data_module.setup()

    train_dataloader = data_module.train_dataloader()

    assert isinstance(train_dataloader, DataLoader)
    assert len(train_dataloader.dataset) > 0
    assert train_dataloader.batch_size == 16
    assert train_dataloader.num_workers == 1
    for data in train_dataloader:
        assert data['input_ids'].shape[0] == 16
        assert data['input_ids'].shape[1] == 128
        assert data['labels'].shape[1] == 128
        assert data['labels'].shape[0] == 16
        break

def test_LMDataModuleGPT2_val_dataloader():
    data_module = LMDataModuleGPT2(
        model_name_or_path="gpt2",
        pad_to_max_length=True,
        preprocessing_num_workers=1,
        overwrite_cache=False,
        max_seq_length=128,
        mlm_probability=0.15,
        train_batch_size=16,
        val_batch_size=16,
        dataloader_num_workers=1,
    )
    data_module.setup()

    val_dataloader = data_module.val_dataloader()

    assert isinstance(val_dataloader, DataLoader)
    assert len(val_dataloader.dataset) > 0
    assert val_dataloader.batch_size == 16
    assert val_dataloader.num_workers == 1
    for data in val_dataloader:
        assert data['input_ids'].shape[0] == 16
        assert data['input_ids'].shape[1] == 128
        break
