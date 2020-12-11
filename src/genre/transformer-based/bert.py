# For running BERT on GPU, see the IPython Notebook.

# Inspired from: https://mccormickml.com/2019/07/22/BERT-fine-tuning/
import torch
import random, time, datetime
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import AdamW, get_linear_schedule_with_warmup

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def prepare_dataloader(texts, labels, IDs=[], batch_size=32, max_length=512):
    """
    Takes as input: texts, labels, and corresponding IDs (in case of test-data)
    This function returns a DataLoader object.

    For train_dataloader, labels are passed. For test_dataloader, both labels and IDs are passed.
    BERT tokenizer is used to
      (1) Tokenize the sentence.
      (2) Prepend the `[CLS]` token to the start.
      (3) Append the `[SEP]` token to the end.
      (4) Map tokens to their IDs.
      (5) Pad or truncate the sentence to `max_length`
      (6) Create attention masks for [PAD] tokens.
    Authors recommend a batch size of 16/32 for fine-tuning.
    """
    input_ids = []; attention_masks = []

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    for sent in texts:
        encoded_dict = tokenizer.encode_plus(sent, # sentence to encode
                                             add_special_tokens=True, # add '[CLS]' and '[SEP]'
                                             truncation=True,
                                             max_length=512,
                                             pad_to_max_length=True,
                                             return_attention_mask=True, # construct attention masks
                                             return_tensors='pt') # return pytorch tensorss


        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask']) # simply differentiates padding from non-padding

    # Convert to tensors:
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    if IDs == []:
        print("Dataset has input_ids, attention_masks, labels")
        dataset = TensorDataset(input_ids, attention_masks, labels)
    else:
        IDs = torch.tensor(IDs)
        print("Dataset has input_ids, attention_masks, labels, and IDs")
        dataset = TensorDataset(input_ids, attention_masks, labels, IDs)

    data_loader = DataLoader(dataset,  # The training samples.
                             sampler=RandomSampler(dataset), # Select batches randomly
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
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

    total_steps = len(data_loader) * epochs # total number of training steps is [number of batches] x [number of epochs]
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    total_t0 = time.time() # keep track of time

    for epoch_i in range(0, epochs):
        print('======== Epoch {:} / {:} ========'.format(epoch_i+1, epochs))
        t0 = time.time()
        total_train_loss = 0 # reset the total loss for this epoch
        model.train() # put the model into training mode

        for batch in data_loader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad() # clears any previously calculated gradients before performing a backward pass

            loss, logits = model(b_input_ids,
                                 token_type_ids=None,
                                 attention_mask=b_input_mask,
                                 labels=b_labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # clip the norm of the gradients to 1.0 to help prevent the "exploding gradients" problem
            optimizer.step() # update parameters and take a step using the computed gradient
            scheduler.step() # update the learning rate

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(data_loader)
        training_time = format_time(time.time() - t0)

        print("\tAverage training loss: {0:.2f}".format(avg_train_loss))
        print("\tTraining epcoh took: {:}".format(training_time))
    print("\n\nTraining complete\nTotal training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

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
