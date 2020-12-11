"""
Author: Sunyam

This file contains the model architecture.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules import FeedForward
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.nn.util import get_text_field_mask
from typing import *


class Classifier(Model):

    def __init__(self, vocab: Vocabulary,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 classifier_feedforward: FeedForward):
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.classifier_feedforward = classifier_feedforward
        self.loss = nn.CrossEntropyLoss()

    # Need to return a dictionary for every forward pass, and compute the loss function within the forward method during training
    def forward(self, tokens: Dict[str, torch.Tensor],
                ID: str, label: torch.Tensor) -> torch.Tensor:
        embeddings = self.word_embeddings(tokens)
        mask = get_text_field_mask(tokens)
        state = self.encoder(embeddings, mask)
        class_logits = self.classifier_feedforward(state)
        class_probabilities = F.softmax(class_logits, dim=-1)

        output = {"class_logits": class_logits,
                  "class_probabilities": class_probabilities
                 }

        output["loss"] = self.loss(class_logits, label) #torch.max(label, 1)[1]) # because CrossEntropyLoss expects class indices, and not one-hot vectors - https://discuss.pytorch.org/t/runtimeerror-multi-target-not-supported-newbie/10216
        return output
