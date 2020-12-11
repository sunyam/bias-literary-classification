# Inspired from: https://mccormickml.com/2019/09/19/XLNet-fine-tuning/
import torch
from transformers import XLNetTokenizer, XLNetForSequenceClassification
from transformers import AdamW
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from tqdm import trange
import random
import numpy as np

def prepare_dataloader(texts, labels, IDs=[], batch_size=32, max_length=512):
    """
    Takes as input: texts, labels, and corresponding IDs (in case of test-data)
    This function returns a DataLoader object.

    For train_dataloader, labels are passed. For test_dataloader, both labels and IDs are passed.
    Uses XLNetTokenizer and does thhe following things:
      - Add the special '[CLS]' and '[SEP]' tokens
      - Tokenize the sentence
      - Map tokens to their IDs
      - Pad or truncate the sentence to 'max_length'
      - Create attention masks
    Authors recommend a batch size of 16/32 for fine-tuning.
    """
    texts_new = [sentence + " [SEP] [CLS]" for sentence in texts] # add special tokens

    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)

    tokenized_texts = [tokenizer.tokenize(sent) for sent in texts_new] # tokenize
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts] # convert to IDs
    input_ids = pad_sequences(input_ids, maxlen=max_length, dtype="long", truncating="post", padding="post") # pad tokens

    # Create attention masks: mask of 1s for each token followed by 0s for padding
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)

    # Convert to tensors:
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    labels = torch.tensor(labels)

    if IDs == []:
        print("Dataset has input_ids, attention_masks, labels")
        dataset = TensorDataset(input_ids, attention_masks, labels)
    else:
        IDs = torch.tensor(IDs)
        print("Dataset has input_ids, attention_masks, labels, and IDs")
        dataset = TensorDataset(input_ids, attention_masks, labels, IDs)

    data_loader = DataLoader(dataset,
                             sampler=RandomSampler(dataset), # select batches randomly
                             batch_size=batch_size)

    print("Input IDs:", input_ids.shape)
    print("Dataset size:", len(dataset))
    return data_loader


def train(data_loader, epochs=3):
    """
    Given the data_loader, it fine-tunes BERT for the specific task.
    The BERT authors recommend between 2 and 4 training epochs.

    Returns fine-tuned BERT model.
    """
    model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased", num_labels=2)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}]

    # This variable contains all of the hyperparemeter information our training loop needs
    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)

    train_loss_set = []

    # trange is a tqdm wrapper around the normal python range
    for _ in trange(epochs, desc="Epoch"):
        model.train()

        # Tracking variables
        tr_loss, nb_tr_examples, nb_tr_steps = 0, 0, 0

        for batch in data_loader:
            batch = tuple(t.to(device) for t in batch)
            optimizer.zero_grad() # clears any previously calculated gradients before performing a backward pass

            b_input_ids, b_input_mask, b_labels = batch
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs[0]
            logits = outputs[1]
            train_loss_set.append(loss.item())

            loss.backward()
            optimizer.step()

            # Update tracking variables
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1

    print("Train loss: {}".format(tr_loss/nb_tr_steps))
    return model

def predict(model, data_loader):
    """
    Given the fine-tuned model and data loader, it returns flat predictions, list of prob(fiction), and corresponding true-labels & IDs.

    For predictions, we pick the label (0 or 1) with the higher score. The output for each batch are a 2-column ndarray (one column for "0"
    and one column for "1"). Pick the label with the highest value and turn this in to a list of 0s and 1s.
    """
    model.eval() # put model in evaluation mode

    predictions, prob_fiction, true_labels, IDs = [], [], [], []

    for batch in data_loader:
        batch = tuple(t.to(device) for t in batch)

        b_input_ids, b_input_mask, b_labels, b_IDs = batch

        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask)

        logits = outputs[0]

        logits = logits.detach().cpu().numpy()
        labels = b_labels.to('cpu').numpy()
        ids = b_IDs.to('cpu').numpy()

        predictions.append(logits)
        true_labels.append(labels)
        IDs.append(ids)


    flat_predictions = np.concatenate(predictions, axis=0)

    probs = torch.nn.functional.softmax(torch.from_numpy(flat_predictions), dim=-1) # convert logits to probabilities
    prob_fiction = probs[:,1] # because order is [0,1] and 1 is fiction
    prob_fiction = prob_fiction.numpy()

    flat_predictions = np.argmax(flat_predictions, axis=1).flatten() # pick the one with the highest value

    flat_true_labels = np.concatenate(true_labels, axis=0)
    flat_IDs = np.concatenate(IDs, axis=0)
    return flat_predictions, prob_fiction, flat_true_labels, flat_IDs


device = 'cpu'
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
