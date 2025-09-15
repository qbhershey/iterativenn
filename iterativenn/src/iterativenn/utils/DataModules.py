import warnings
from itertools import chain

import pytorch_lightning as pl
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import Dataset
from transformers import (AutoConfig, AutoTokenizer, DataCollatorForLanguageModeling, default_data_collator)
from iterativenn.nn_modules.nlp import BertEmbeddings
from iterativenn.utils.DatasetUtils import CustomTensorDataset, ImageSequence

class SequenceModule(pl.LightningDataModule):
    def __init__(self, 
                 dataset: Dataset,
                 batch_size: int = 32,
                 train_size: int = 128, 
                 val_size: int = 128,
                 test_size: int = 128,
                 num_workers: int = 4,
                 seed: int = None):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.num_workers = num_workers

    def trivial_collate_batch(self, batch):
        return batch

    def train_dataloader(self):
        return DataLoader(self.dataset(size=self.train_size), num_workers=self.num_workers, batch_size=self.batch_size, collate_fn=self.trivial_collate_batch)   

    def val_dataloader(self):
        return DataLoader(self.dataset(size=self.val_size), num_workers=self.num_workers, batch_size=self.batch_size, collate_fn=self.trivial_collate_batch)   

    def test_dataloader(self):
        return DataLoader(self.dataset(size=self.test_size), num_workers=self.num_workers, batch_size=self.batch_size, collate_fn=self.trivial_collate_batch)   

    def predict_dataloader(self):
        return DataLoader(self.dataset(size=self.test_size), num_workers=self.num_workers, batch_size=self.batch_size, collate_fn=self.trivial_collate_batch)

class MNISTRepeatedSequenceDataModule(pl.LightningDataModule):
    def __init__(self, 
                 batch_size: int = 32,
                 train_size: int = 128, 
                 val_size: int = 128,
                 min_copies: int = 1,
                 max_copies: int = 1,
                 data_dir: str = '~/work/data',
                 seed: int = None):
        """
        This is based on the example at
        https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/datamodules.html

        Args:
            batch_size (int, optional): The batch size between each gradient update. Defaults to 32.
            train_size (int, optional): The number of training images to use. Defaults to 100.
            val_size (int, optional): The number of validation images to use. Defaults to 100.
            min_copies (int, optional): The mininum number of copies of the image to put into the sequence. Defaults to 1.
            max_copies (int, optional): The maximum number of copies of the image to put into the sequence. Defaults to 1.
            data_dir (str, optional): The directory in which to store the data. Defaults to '~/work/data'.
        """

        super().__init__()
        self.batch_size = batch_size
        self.train_size = train_size
        self.val_size = val_size
        self.min_copies = min_copies
        self.max_copies = max_copies
        self.data_dir = data_dir
        # self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.seed = seed

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        # IMPORTANT:  You need to be careful here.  There are places where the training and validation datasets can be reshuffled, and this
        # is a place where that happens.  I.e., "setup" is called multiple times, and if you don't fix the seed, you will get different.  
        # In particular, this happens when you call "trainer.fit" multiple times. 
        if self.seed is not None:
            torch.manual_seed(self.seed)
        if stage == "fit":
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val, _ = random_split(mnist_full, [self.train_size, self.val_size, 60000 - self.train_size - self.val_size])

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)
            
        if stage == "predict":
            self.mnist_predict = MNIST(self.data_dir, train=False, transform=self.transform)

    def get_dataloader_(self, dataset):
        """ This is a helper function to create a dataloader for a given dataset.  It should be all that you
            need to change if you want to change the sequence generator.
        """    
        # The default collate_fn trys to do smart things about combining a minibatch into a single
        # matrix.  This is likely not what we want in this general case, though there are things we can 
        # learn from what other people do.  For example,

        # https://androidkt.com/create-dataloader-with-collate_fn-for-variable-length-input-in-pytorch/

        # talks about all of this in some detail, in the case of NLP.  The idea is that variable length data
        # can be "zero padded", and there is even a pytorch helper function for this very case.

        #  https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_sequence.html
        def trivial_collate_batch(batch):
            return batch

        # Turn the images into image sequences
        # Note, this is where we get the number of iterations for the model!
        # I have messed this up in the past, and I need to be careful about it.
        # max_copies=1 makes this the same as the previous example.  I.e.,
        # the map is an MLP, and not an INN,
        dataset = ImageSequence(dataset, min_copies=self.min_copies, max_copies=self.max_copies, evaluate_loss='last')
        return DataLoader(dataset, batch_size=self.batch_size, collate_fn=trivial_collate_batch)

    def train_dataloader(self):
        return self.get_dataloader_(self.mnist_train)

    def val_dataloader(self):
        return self.get_dataloader_(self.mnist_val)

    def test_dataloader(self):
        return self.get_dataloader_(self.mnist_test)

    def predict_dataloader(self):
        return self.get_dataloader_(self.mnist_predict)


# Adapted from https://github.com/huggingface/transformers/blob/master/examples/language-modeling/run_mlm.py
class LMDataModule(pl.LightningDataModule):
    def __init__(self, model_name_or_path, line_by_line, pad_to_max_length,
                 preprocessing_num_workers, overwrite_cache, max_seq_length, mlm_probability,
                 train_batch_size, val_batch_size, dataloader_num_workers):
        super().__init__()
        #self.train_file = train_file
        #self.validation_file = validation_file
        self.model_name_or_path = model_name_or_path
        self.line_by_line = line_by_line
        self.pad_to_max_length = pad_to_max_length
        self.preprocessing_num_workers = preprocessing_num_workers
        self.overwrite_cache = overwrite_cache
        self.max_seq_length = max_seq_length
        self.mlm_probability = mlm_probability
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.dataloader_num_workers = dataloader_num_workers
        self.config = AutoConfig.from_pretrained(
            "bert-base-cased", return_dict=True)
        self.bert = BertEmbeddings(config=self.config)

    def setup(self,  stage=None):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        #extension = self.train_file.split(".")[-1]
        #if extension in ("txt", "raw"):
        #    extension = "text"

        data_files = {}
        #data_files["train"] = self.train_file
        #data_files["validation"] = self.validation_file
        datasets = load_dataset('wikitext', 'wikitext-2-raw-v1')
        #datasets = load_dataset(extension, data_files=data_files)

        column_names = datasets["train"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]

        if self.line_by_line:
            # When using line_by_line, we just tokenize each nonempty line.
            padding = "max_length" if self.pad_to_max_length else False

            def tokenize_function(examples):
                # Remove empty lines
                examples["text"] = [line for line in examples["text"]
                                    if len(line) > 0 and not line.isspace()]
                return tokenizer(
                    examples["text"],
                    padding=padding,
                    truncation=True,
                    max_length=self.max_seq_length,
                    # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                    # receives the `special_tokens_mask`.
                    return_special_tokens_mask=True,
                )

            tokenized_datasets = datasets.map(
                tokenize_function,
                batched=True,
                num_proc=self.preprocessing_num_workers,
                remove_columns=[text_column_name],
                load_from_cache_file=not self.overwrite_cache,
            )
        else:
            # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
            # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
            # efficient when it receives the `special_tokens_mask`.
            def tokenize_function(examples):
                return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

            tokenized_datasets = datasets.map(
                tokenize_function,
                batched=True,
                num_proc=self.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not self.overwrite_cache,
            )

            if self.max_seq_length is None:
                self.max_seq_length = tokenizer.model_max_length
            else:
                if self.max_seq_length > tokenizer.model_max_length:
                    warnings.warn(
                        f"The max_seq_length passed ({self.max_seq_length}) is larger than the maximum length for the"
                        f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
                    )
                self.max_seq_length = min(self.max_seq_length, tokenizer.model_max_length)

            # Main data processing function that will concatenate all texts from our dataset and generate chunks of
            # max_seq_length.
            def group_texts(examples):
                # Concatenate all texts.
                concatenated_examples = {
                    k: sum(examples[k], []) for k in examples.keys()}
                total_length = len(concatenated_examples[list(examples.keys())[0]])
                # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
                # customize this part to your needs.
                total_length = (total_length // self.max_seq_length) * self.max_seq_length
                # Split by chunks of max_len.
                result = {
                    k: [t[i: i + self.max_seq_length]
                        for i in range(0, total_length, self.max_seq_length)]
                    for k, t in concatenated_examples.items()
                }
                return result

            # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
            # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
            # might be slower to preprocess.
            #
            # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
            tokenized_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=self.preprocessing_num_workers,
                load_from_cache_file=not self.overwrite_cache,
            )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm_probability=self.mlm_probability)


        train_dataset = tokenized_datasets["validation"]
        eval_dataset = tokenized_datasets["validation"]


        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator

    @staticmethod
    def flatten_input(input_tensor):
        n_batch, batch_size, seq_len, input_dim = input_tensor.shape
        output_tensor = input_tensor.view(n_batch*batch_size, seq_len, input_dim)
        return output_tensor

    @staticmethod
    def flatten_input_y(input_tensor):
        n_batch, batch_size, seq_len = input_tensor.shape
        output_tensor = input_tensor.view(n_batch * batch_size, seq_len)
        return output_tensor

    def train_dataloader(self):
        data_x = []
        data_y = []

        dl = DataLoader(
            self.eval_dataset,
            batch_size=self.train_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.dataloader_num_workers,
        )

        with torch.no_grad():
            for batch in dl:
                x, y = self.bert(batch)
                if len(x) == self.train_batch_size and len(y) == self.train_batch_size:
                    data_x.append(x)
                    data_y.append(y)
        #out_tensor_x = torch.zeros((dl.__len__(), x.shape[0], x.shape[1], x.shape[2]), requires_grad=False).float()
        #out_tensor_y = torch.zeros((dl.__len__(), x.shape[0], x.shape[1]), requires_grad=False).float()
        data_x_t = self.flatten_input(torch.stack(data_x).float())
        data_y_t = self.flatten_input_y(torch.stack(data_y).float())

        return DataLoader(
            CustomTensorDataset(data_x_t, data_y_t),
            batch_size=self.train_batch_size,
            collate_fn=None,
            num_workers=self.dataloader_num_workers,
        )



    def val_dataloader(self):
        data_x = []
        data_y = []

        dl = DataLoader(
            self.eval_dataset,
            batch_size=self.val_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.dataloader_num_workers,
        )


        with torch.no_grad():
            for batch in dl:
                x, y = self.bert(batch)
                if len(x) == self.val_batch_size and len(y) == self.val_batch_size:
                    data_x.append(x)
                    data_y.append(y)

        #out_tensor_x = torch.zeros((dl.__len__(), x.shape[0], x.shape[1], x.shape[2]), requires_grad=True).float()
        #out_tensor_y = torch.zeros((dl.__len__(), x.shape[0], x.shape[1]), requires_grad=True).float()
        data_x_t = self.flatten_input(torch.stack(data_x).float())
        data_y_t = self.flatten_input_y(torch.stack(data_y).float())

        return DataLoader(
                CustomTensorDataset(data_x_t, data_y_t),
                batch_size=self.val_batch_size,
                collate_fn=None,
                num_workers=self.dataloader_num_workers,
            )

# Adapted from https://github.com/huggingface/transformers/blob/master/examples/language-modeling/run_mlm.py
class LMDataModuleGPT2(pl.LightningDataModule):
    def __init__(self, model_name_or_path, pad_to_max_length,
                 preprocessing_num_workers, overwrite_cache, max_seq_length, mlm_probability,
                 train_batch_size, val_batch_size, dataloader_num_workers):
        super().__init__()
        #self.train_file = train_file
        #self.validation_file = validation_file
        self.model_name_or_path = model_name_or_path
        self.pad_to_max_length = pad_to_max_length
        self.preprocessing_num_workers = preprocessing_num_workers
        self.overwrite_cache = overwrite_cache
        self.max_seq_length = max_seq_length
        self.mlm_probability = mlm_probability
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.dataloader_num_workers = dataloader_num_workers


    def setup(self,  stage=None):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        #tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        #extension = self.train_file.split(".")[-1]
        #if extension in ("txt", "raw"):
        #    extension = "text"

        #data_files = {}
        #data_files["train"] = self.train_file
        #data_files["validation"] = self.validation_file
        datasets = load_dataset('wikitext', 'wikitext-2-raw-v1')
        #datasets = load_dataset(extension, data_files=data_files)

        column_names = datasets["train"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]

        def tokenize_function(examples):
            return tokenizer(examples[text_column_name])

        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            num_proc=self.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not self.overwrite_cache,
        )

        if self.max_seq_length is None:
            self.max_seq_length = tokenizer.model_max_length
        else:
            if self.max_seq_length > tokenizer.model_max_length:
                warnings.warn(
                    f"The max_seq_length passed ({self.max_seq_length}) is larger than the maximum length for the"
                    f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
                )
            self.max_seq_length = min(self.max_seq_length, tokenizer.model_max_length)

            # Main data processing function that will concatenate all texts from our dataset and generate chunks of
            # max_seq_length.
            def group_texts(examples):
                # Concatenate all texts.
                concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}

                total_length = len(concatenated_examples[list(examples.keys())[0]])
                # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
                # customize this part to your needs.
                if total_length >= self.max_seq_length:
                    total_length = (total_length // self.max_seq_length) * self.max_seq_length
                # Split by chunks of max_len.
                result = {
                    k: [t[i: i + self.max_seq_length]
                        for i in range(0, total_length, self.max_seq_length)]
                    for k, t in concatenated_examples.items()
                }
                result["labels"] = result["input_ids"].copy()
                return result

            # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
            # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
            # might be slower to preprocess.
            #
            # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
            tokenized_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=self.preprocessing_num_workers,
                load_from_cache_file=not self.overwrite_cache,
            )
            self.train_dataset = tokenized_datasets["validation"]
            self.eval_dataset = tokenized_datasets["test"]


    def train_dataloader(self):
       return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            collate_fn=default_data_collator,
            num_workers=self.dataloader_num_workers,
           shuffle=False,
           drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.eval_dataset,
            batch_size=self.val_batch_size,
            collate_fn=default_data_collator,
            num_workers=self.dataloader_num_workers,
            shuffle=False,
            drop_last=True,
        )
    def predict_dataloader(self) :
        return DataLoader(
            self.eval_dataset,
            batch_size=self.val_batch_size,
            collate_fn=default_data_collator,
            num_workers=self.dataloader_num_workers,
            shuffle=False,
            drop_last=True,
        )
