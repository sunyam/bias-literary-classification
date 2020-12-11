"""
This file is for reading the Novel dataset as AllenNLP Instances.
"""

from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, MetadataField, LabelField #, ArrayField
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data import Instance
from allennlp.data.tokenizers import Token

from typing import *
from overrides import overrides
import numpy as np
import os
import sys
sys.path.append('/path/Augmentation-for-Literary-Data/src/genre-bias/traditional-machine-learning/')
import data_loader


class NovelDatasetReader(DatasetReader):
    """
    Reads the Past dataset. When reading, takes as input the file path of the folder.
    """
    def __init__(self, scenario: str, augmentation: str=None,
                 tokenizer: Callable[[str], List[str]]=lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer]=None) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers
        self.scenario = scenario
        self.augmentation = augmentation

    @overrides
    def text_to_instance(self, tokens: List[Token],
                         ID: str=None,
                         label: str=None) -> Instance:
        text = TextField(tokens, self.token_indexers)
        fields = {"tokens": text}

        id_field = MetadataField(ID)
        fields["ID"] = id_field

        label_field = LabelField(label)
        fields["label"] = label_field

        return Instance(fields)

    @overrides
    def _read(self, file_path: str) -> Iterator[Instance]:
        if self.augmentation == None:
            train_loader_fn = data_loader.load_train_data
        elif self.augmentation == 'EDA':
            train_loader_fn = data_loader.load_train_data_with_EDA
        elif self.augmentation == 'CDA':
            pass
        else:
            print("Invalid augmentation:", self.augmentation)


        if file_path == 'train':
            X, Y = train_loader_fn(self.scenario)
            X = X.tolist(); Y = Y.tolist()
            for text, label in zip(X, Y):
                yield self.text_to_instance(
                    tokens=[Token(x) for x in self.tokenizer(text)],
                    ID='__', # ID for every row
                    label=label
                )

        elif file_path == 'test':
            X, Y, IDs = data_loader.load_test_data()
            X = X.tolist(); Y = Y.tolist()
            for text, label, ID in zip(X, Y, IDs):
                yield self.text_to_instance(
                    tokens=[Token(x) for x in self.tokenizer(text)],
                    ID=ID, # unique ID for every test row
                    label=label
                )

        else:
            print("Invalid split parameter:", data)
