# %% [markdown]
# # LSTM-arithmetic
# 
# ## Dataset
# - [Arithmetic dataset](https://drive.google.com/file/d/1cMuL3hF9jefka9RyF4gEBIGGeFGZYHE-/view?usp=sharing)

# %%
#from google.colab import drive
#drive.mount('/content/gdrive')

# %%
#! pip install opencc

# %%
# ! pip install seaborn
# ! pip install opencc
# ! pip install -U scikit-learn

import numpy as np
import pandas as pd
import torch
import torch.nn
import torch.nn.utils.rnn
import torch.utils.data
import matplotlib.pyplot as plt
import seaborn as sns
import opencc
import os
from sklearn.model_selection import train_test_split

#data_path = './data'
data_path = './gdrive/MyDrive/NLP/HW2/'

# %%
#df_train = pd.read_csv(os.path.join(data_path, 'arithmetic_train.csv'))
#df_eval = pd.read_csv(os.path.join(data_path, 'arithmetic_eval.csv'))
#df_train.head()

# %%
df_train = pd.read_csv(os.path.join('arithmetic_train.csv'))
df_eval = pd.read_csv(os.path.join('arithmetic_eval.csv'))

# %%
df_train = df_train.iloc[:, 1:]
df_eval = df_eval.iloc[:, 1:]

# %%
print(df_train.head())
print(df_eval.head())

# %%
# transform the input data to string
df_train['tgt'] = df_train['tgt'].apply(lambda x: str(x))
df_train['src'] = df_train['src'].add(df_train['tgt'])
df_train['len'] = df_train['src'].apply(lambda x: len(x))

df_eval['tgt'] = df_eval['tgt'].apply(lambda x: str(x))
df_eval['src'] = df_eval['src'].add(df_eval['tgt'])
df_eval['len'] = df_eval['src'].apply(lambda x: len(x))

# %%
df_eval.head()

# %% [markdown]
# # Build Dictionary
#  - The model cannot perform calculations directly with plain text.
#  - Convert all text (numbers/symbols) into numerical representations.
#  - Special tokens
#     - '&lt;pad&gt;'
#         - Each sentence within a batch may have different lengths.
#         - The length is padded with '&lt;pad&gt;' to match the longest sentence in the batch.
#     - '&lt;eos&gt;'
#         - Specifies the end of the generated sequence.
#         - Without '&lt;eos&gt;', the model will not know when to stop generating.

# %%
char_to_id = {}
id_to_char = {}

# write your code here
special_tokens = ['<pad>', '<eos>']
# find every letter in src and make a set with pad and eos
unique_chars = set(''.join(df_train['src'].tolist()))
all_tokens = special_tokens + sorted(unique_chars)
# enumerate every elements in the set and create two dictionary
char_to_id = {char: idx for idx, char in enumerate(all_tokens)}
id_to_char = {idx: char for char, idx in char_to_id.items()}
# Build a dictionary and give every token in the train dataset an id
# The dictionary should contain <eos> and <pad>
# char_to_id is to conver charactors to ids, while id_to_char is the opposite

vocab_size = len(char_to_id)
print('Vocab size{}'.format(vocab_size))

# %%
print(char_to_id)
print(id_to_char)

# %% [markdown]
# # Data Preprocessing
#  - The data is processed into the format required for the model's input and output.
#  - Example: 1+2-3=0
#      - Model input: 1 + 2 - 3 = 0
#      - Model output: / / / / / 0 &lt;eos&gt;  (the '/' can be replaced with &lt;pad&gt;)
#      - The key for the model's output is that the model does not need to predict the next character of the previous part. What matters is that once the model sees '=', it should start generating the answer, which is '0'. After generating the answer, it should also generate&lt;eos&gt;
# 

# %%
# Write your code here


def preprocess_data(df, char_to_id):
    char_id_lists = []
    label_id_lists = []
    # process data row by row
    for index, row in df.iterrows():
        equation = row['src']
        target = str(row['tgt'])

        # convert string to id by our dictionary char_to_id
        char_id_list = [char_to_id[char] for char in equation]

        # same to label_id_list but fill with pad for same length
        label_id_list = [char_to_id['<pad>']] * len(char_id_list)

        # everything on the right of = must be our target
        equal_idx = equation.index('=') + 1

        # add it to the label_id_list
        for idx, char in enumerate(target):
            label_id_list[equal_idx + idx] = char_to_id[char]

        # append eos for end of string
        if equal_idx + len(target) < len(label_id_list):
            label_id_list[equal_idx + len(target)] = char_to_id['<eos>']
        else:
            label_id_list.append(char_to_id['<eos>'])

        if equal_idx + len(target) < len(char_id_list):
            char_id_list[equal_idx + len(target)] = char_to_id['<eos>']
        else:
            char_id_list.append(char_to_id['<eos>'])

        # append result to the dataframe
        char_id_lists.append(char_id_list)
        label_id_lists.append(label_id_list)


    df['char_id_list'] = char_id_lists
    df['label_id_list'] = label_id_lists

# apply
preprocess_data(df_train, char_to_id)

df_train.head()

# %%
preprocess_data(df_eval, char_to_id)

df_eval.head()

# %%
#x = df_train.iloc[1, 3][:df_train.iloc[0, 3].index(17) + 1]# Write your code here
#y = df_train.iloc[1, 4][-len(x):]
#print(x, y)

# %% [markdown]
# # Hyper Parameters
# 
# |Hyperparameter|Meaning|Value|
# |-|-|-|
# |`batch_size`|Number of data samples in a single batch|64|
# |`epochs`|Total number of epochs to train|10|
# |`embed_dim`|Dimension of the word embeddings|256|
# |`hidden_dim`|Dimension of the hidden state in each timestep of the LSTM|256|
# |`lr`|Learning Rate|0.001|
# |`grad_clip`|To prevent gradient explosion in RNNs, restrict the gradient range|1|

# %%
batch_size = 64
epochs = 2
embed_dim = 256
hidden_dim = 256
lr = 0.001
grad_clip = 1

# %% [markdown]
# # Data Batching
# - Use `torch.utils.data.Dataset` to create a data generation tool called  `dataset`.
# - The, use `torch.utils.data.DataLoader` to randomly sample from the `dataset` and group the samples into batches.

# %%
class Dataset(torch.utils.data.Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        # return the amount of data
        return len(self.sequences)# Write your code here

    def __getitem__(self, index):
        # Extract the input data x and the ground truth y from the data
        x = self.sequences.iloc[index, 0][:-1]# Write your code here
        y = self.sequences.iloc[index, 1][1:len(x)+1]# Write your code here
        return x, y

# collate function, used to build dataloader
def collate_fn(batch):
    batch_x = [torch.tensor(data[0]) for data in batch]
    batch_y = [torch.tensor(data[1]) for data in batch]
    batch_x_lens = torch.LongTensor([len(x) for x in batch_x])
    batch_y_lens = torch.LongTensor([len(y) for y in batch_y])

    # Pad the input sequence
    pad_batch_x = torch.nn.utils.rnn.pad_sequence(batch_x,
                                                  batch_first=True,
                                                  padding_value=char_to_id['<pad>'])

    pad_batch_y = torch.nn.utils.rnn.pad_sequence(batch_y,
                                                  batch_first=True,
                                                  padding_value=char_to_id['<pad>'])

    return pad_batch_x, pad_batch_y, batch_x_lens, batch_y_lens

# %%
ds_train = Dataset(df_train[['char_id_list', 'label_id_list']])
ds_eval = Dataset(df_eval[['char_id_list', 'label_id_list']])

# %%
#from torch.utils.data import DataLoader

# %%
# Build dataloader of train set and eval set, collate_fn is the collate function
dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)# Write your code here
dl_eval = torch.utils.data.DataLoader(ds_eval, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)# Write your code here

# %% [markdown]
# # Model Design
# 
# ## Execution Flow
# 1. Convert all characters in the sentence into embeddings.
# 2. Pass the embeddings through an LSTM sequentially.
# 3. The output of the LSTM is passed into another LSTM, and additional layers can be added.
# 4. The output from all time steps of the final LSTM is passed through a Fully Connected layer.
# 5. The character corresponding to the maximum value across all output dimensions is selected as the next character.
# 
# ## Loss Function
# Since this is a classification task, Cross Entropy is used as the loss function.
# 
# ## Gradient Update
# Adam algorithm is used for gradient updates.

# %%
class CharRNN(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(CharRNN, self).__init__()

        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size,
                                            embedding_dim=embed_dim,
                                            padding_idx=char_to_id['<pad>'])

        self.rnn_layer1 = torch.nn.LSTM(input_size=embed_dim,
                                        hidden_size=hidden_dim,
                                        batch_first=True)

        self.rnn_layer2 = torch.nn.LSTM(input_size=hidden_dim,
                                        hidden_size=hidden_dim,
                                        batch_first=True)

        self.linear = torch.nn.Sequential(torch.nn.Linear(in_features=hidden_dim,
                                                          out_features=hidden_dim),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(in_features=hidden_dim,
                                                          out_features=vocab_size))

    def forward(self, batch_x, batch_x_lens):
        return self.encoder(batch_x, batch_x_lens)

    # The forward pass of the model
    def encoder(self, batch_x, batch_x_lens):
        batch_x = self.embedding(batch_x)

        batch_x = torch.nn.utils.rnn.pack_padded_sequence(batch_x,
                                                          batch_x_lens,
                                                          batch_first=True,
                                                          enforce_sorted=False)

        batch_x, _ = self.rnn_layer1(batch_x)
        batch_x, _ = self.rnn_layer2(batch_x)

        batch_x, _ = torch.nn.utils.rnn.pad_packed_sequence(batch_x,
                                                            batch_first=True)

        batch_x = self.linear(batch_x)

        return batch_x

    def generator(self, start_char, max_len=200):

        char_list = [char_to_id[c] for c in start_char]

        next_char = None
        device = next(self.parameters()).device

        while len(char_list) < max_len:
            # Write your code here
            input_tensor = torch.tensor(char_list).unsqueeze(0).to(device)

            # Pass through the embedding layer
            embedded = self.embedding(input_tensor)

            # Pass through the LSTM layers

            output, _ = self.rnn_layer1(embedded)
            output, _ = self.rnn_layer2(output)

            # Pass through the linear layers to get logits
            logits = self.linear(output)
            # Pack the char_list to tensor
            # Input the tensor to the embedding layer, LSTM layers, linear respectively
            y = logits[:, -1, :]# Obtain the next token prediction y

            next_char = torch.argmax(y, dim=-1).item()# Use argmax function to get the next token prediction

            if next_char == char_to_id['<eos>']:
                break

            char_list.append(next_char)

        return [id_to_char[ch_id] for ch_id in char_list]

# %%
torch.manual_seed(2)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")# Write your code here. Specify a device (cuda or cpu)

model = CharRNN(vocab_size,
                embed_dim,
                hidden_dim)

# %%
criterion = torch.nn.CrossEntropyLoss(ignore_index=char_to_id['<pad>'])# Write your code here. Cross-entropy loss function. The loss function should ignore <pad>
optimizer = torch.optim.Adam(model.parameters(), lr=lr)# Write your code here. Use Adam or AdamW for Optimizer

# %% [markdown]
# # Training
# 1. The outer `for` loop controls the `epoch`
#     1. The inner `for` loop uses `data_loader` to retrieve batches.
#         1. Pass the batch to the `model` for training.
#         2. Compare the predicted results `batch_pred_y` with the true labels `batch_y` using Cross Entropy to calculate the loss `loss`
#         3. Use `loss.backward` to automatically compute the gradients.
#         4. Use `torch.nn.utils.clip_grad_value_` to limit the gradient values between `-grad_clip` &lt; and &lt; `grad_clip`.
#         5. Use `optimizer.step()` to update the model (backpropagation).
# 2.  After every `1000` batches, output the current loss to monitor whether it is converging.

# %%
from tqdm import tqdm
from copy import deepcopy
model = model.to(device)
model.train()
i = 0
for epoch in range(1, epochs+1):
    # The process bar
    bar = tqdm(dl_train, desc=f"Train epoch {epoch}")
    for batch_x, batch_y, batch_x_lens, batch_y_lens in bar:
        # Write your code here
        # Clear the gradient
        optimizer.zero_grad()
        batch_pred_y = model(batch_x.to(device), batch_x_lens)

        # Write your code here
        batch_y = batch_y.to(device)
        # Input the prediction and ground truths to loss function
        loss = criterion(batch_pred_y.view(-1, vocab_size), batch_y.view(-1))
        # Back propagation
        loss.backward()

        torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip) # gradient clipping

        # Write your code here
        # Optimize parameters in the model
        optimizer.step()

        i+=1
        if i%50==0:
            bar.set_postfix(loss = loss.item())

    # Evaluate your model
    model.eval()

    bar = tqdm(dl_eval, desc=f"Validation epoch {epoch}")
    matched = 0
    total = 0

    for batch_x, batch_y, batch_x_lens, batch_y_lens in bar:
        batch_x = batch_x.to(device)

        predictions = [] # Write your code here. Input the batch_x to the model and generate the predictions
        for x in batch_x:
            #input_sequence = ''.join([id_to_char[idx] for idx in x.tolist() if idx in id_to_char])
            input_sequence = ''.join([id_to_char[idx] for idx in x.tolist() if idx in id_to_char and id_to_char[idx] != '<pad>'])
            #input_sequence = ''.join([id_to_char[idx] if id_to_char[idx] != '<pad>' else '0' for idx in x.tolist() if idx in id_to_char])
            pred = model.generator(input_sequence)
            pred_ids = [char_to_id[char] for char in pred if char in char_to_id]

            if len(pred_ids) < batch_y.size(1):
                pred_ids += [char_to_id['<pad>']] * (batch_y.size(1) - len(pred_ids))
            predictions.append(pred_ids)

        predictions_tensor = torch.tensor(predictions, dtype=torch.int64).to(device)
        predictions_left = predictions_tensor
        #predictions_tensor[:, -1] = char_to_id['<pad>']
        predictions_left[::, 0:-1] = predictions_tensor[::, 1:]
        predictions_left[::, -1] = char_to_id['<pad>']
        batch_y = batch_y.to(device)
        # Write your code here.
        # Check whether the prediction match the ground truths
        # Compute exact match (EM) on the eval dataset
        # EM = correct/total
        for pred, true in zip(predictions_left, batch_y):
            mask = (true != char_to_id['<pad>']) & (true != char_to_id['<eos>'])
            if torch.equal(pred[mask], true[mask]):
                matched += 1
            total += 1


    print(matched/total)
    model.train()

# %% [markdown]
# # Generation
# Use `model.generator` and provide an initial character to automatically generate a sequence.

# %%
model = model.to("cpu")
print("".join(model.generator('1+1=')))

# %% [markdown]
# # Test #

# %% [markdown]
# chaange learning rate

# %%
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# %%
model = model.to(device)
model.train()
i = 0
for epoch in range(1, epochs+1):
    # The process bar
    bar = tqdm(dl_train, desc=f"Train epoch {epoch}")
    for batch_x, batch_y, batch_x_lens, batch_y_lens in bar:
        # Write your code here
        # Clear the gradient
        optimizer.zero_grad()
        batch_pred_y = model(batch_x.to(device), batch_x_lens)

        # Write your code here
        batch_y = batch_y.to(device)
        # Input the prediction and ground truths to loss function
        loss = criterion(batch_pred_y.view(-1, vocab_size), batch_y.view(-1))
        # Back propagation
        loss.backward()

        torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip) # gradient clipping

        # Write your code here
        # Optimize parameters in the model
        optimizer.step()

        i+=1
        if i%50==0:
            bar.set_postfix(loss = loss.item())

    # Evaluate your model
    model.eval()

    bar = tqdm(dl_eval, desc=f"Validation epoch {epoch}")
    matched = 0
    total = 0

    for batch_x, batch_y, batch_x_lens, batch_y_lens in bar:
        batch_x = batch_x.to(device)

        predictions = [] # Write your code here. Input the batch_x to the model and generate the predictions
        for x in batch_x:
            #input_sequence = ''.join([id_to_char[idx] for idx in x.tolist() if idx in id_to_char])
            input_sequence = ''.join([id_to_char[idx] for idx in x.tolist() if idx in id_to_char and id_to_char[idx] != '<pad>'])
            #input_sequence = ''.join([id_to_char[idx] if id_to_char[idx] != '<pad>' else '0' for idx in x.tolist() if idx in id_to_char])
            pred = model.generator(input_sequence)
            pred_ids = [char_to_id[char] for char in pred if char in char_to_id]

            if len(pred_ids) < batch_y.size(1):
                pred_ids += [char_to_id['<pad>']] * (batch_y.size(1) - len(pred_ids))
            predictions.append(pred_ids)

        predictions_tensor = torch.tensor(predictions, dtype=torch.int64).to(device)
        predictions_left = predictions_tensor
        #predictions_tensor[:, -1] = char_to_id['<pad>']
        predictions_left[::, 0:-1] = predictions_tensor[::, 1:]
        predictions_left[::, -1] = char_to_id['<pad>']
        batch_y = batch_y.to(device)
        # Write your code here.
        # Check whether the prediction match the ground truths
        # Compute exact match (EM) on the eval dataset
        # EM = correct/total
        for pred, true in zip(predictions_left, batch_y):
            mask = (true != char_to_id['<pad>']) & (true != char_to_id['<eos>'])
            if torch.equal(pred[mask], true[mask]):
                matched += 1
            total += 1


    print(matched/total)
    model.train()

# %% [markdown]
# Without gradient clipping

# %%
model = model.to(device)
model.train()
i = 0
for epoch in range(1, epochs+1):
    # The process bar
    bar = tqdm(dl_train, desc=f"Train epoch {epoch}")
    for batch_x, batch_y, batch_x_lens, batch_y_lens in bar:
        # Write your code here
        # Clear the gradient
        optimizer.zero_grad()
        batch_pred_y = model(batch_x.to(device), batch_x_lens)

        # Write your code here
        batch_y = batch_y.to(device)
        # Input the prediction and ground truths to loss function
        loss = criterion(batch_pred_y.view(-1, vocab_size), batch_y.view(-1))
        # Back propagation
        loss.backward()

        #torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip) # gradient clipping

        # Write your code here
        # Optimize parameters in the model
        optimizer.step()

        i+=1
        if i%50==0:
            bar.set_postfix(loss = loss.item())

    # Evaluate your model
    model.eval()

    bar = tqdm(dl_eval, desc=f"Validation epoch {epoch}")
    matched = 0
    total = 0

    for batch_x, batch_y, batch_x_lens, batch_y_lens in bar:
        batch_x = batch_x.to(device)

        predictions = [] # Write your code here. Input the batch_x to the model and generate the predictions
        for x in batch_x:
            #input_sequence = ''.join([id_to_char[idx] for idx in x.tolist() if idx in id_to_char])
            input_sequence = ''.join([id_to_char[idx] for idx in x.tolist() if idx in id_to_char and id_to_char[idx] != '<pad>'])
            #input_sequence = ''.join([id_to_char[idx] if id_to_char[idx] != '<pad>' else '0' for idx in x.tolist() if idx in id_to_char])
            pred = model.generator(input_sequence)
            pred_ids = [char_to_id[char] for char in pred if char in char_to_id]

            if len(pred_ids) < batch_y.size(1):
                pred_ids += [char_to_id['<pad>']] * (batch_y.size(1) - len(pred_ids))
            predictions.append(pred_ids)

        predictions_tensor = torch.tensor(predictions, dtype=torch.int64).to(device)
        predictions_left = predictions_tensor
        #predictions_tensor[:, -1] = char_to_id['<pad>']
        predictions_left[::, 0:-1] = predictions_tensor[::, 1:]
        predictions_left[::, -1] = char_to_id['<pad>']
        batch_y = batch_y.to(device)
        # Write your code here.
        # Check whether the prediction match the ground truths
        # Compute exact match (EM) on the eval dataset
        # EM = correct/total
        for pred, true in zip(predictions_left, batch_y):
            mask = (true != char_to_id['<pad>']) & (true != char_to_id['<eos>'])
            if torch.equal(pred[mask], true[mask]):
                matched += 1
            total += 1


    print(matched/total)
    model.train()

# %% [markdown]
# Use GRU instead of LSTM

# %%
class CharRNN(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(CharRNN, self).__init__()

        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size,
                                            embedding_dim=embed_dim,
                                            padding_idx=char_to_id['<pad>'])

        # 使用 GRU 代替 LSTM
        self.rnn_layer1 = torch.nn.GRU(input_size=embed_dim,
                                       hidden_size=hidden_dim,
                                       batch_first=True)

        self.rnn_layer2 = torch.nn.GRU(input_size=hidden_dim,
                                       hidden_size=hidden_dim,
                                       batch_first=True)

        self.linear = torch.nn.Sequential(
            torch.nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=hidden_dim, out_features=vocab_size)
        )

    def forward(self, batch_x, batch_x_lens):
        return self.encoder(batch_x, batch_x_lens)

    # The forward pass of the model
    def encoder(self, batch_x, batch_x_lens):
        batch_x = self.embedding(batch_x)

        # Pack the input sequences
        batch_x = torch.nn.utils.rnn.pack_padded_sequence(batch_x,
                                                          batch_x_lens,
                                                          batch_first=True,
                                                          enforce_sorted=False)

        batch_x, _ = self.rnn_layer1(batch_x)
        batch_x, _ = self.rnn_layer2(batch_x)

        # Unpack the sequences
        batch_x, _ = torch.nn.utils.rnn.pad_packed_sequence(batch_x, batch_first=True)

        batch_x = self.linear(batch_x)
        return batch_x
    
    def generator(self, start_char, max_len=200):

        char_list = [char_to_id[c] for c in start_char]

        next_char = None
        device = next(self.parameters()).device

        while len(char_list) < max_len:
            # Write your code here
            input_tensor = torch.tensor(char_list).unsqueeze(0).to(device)

            # Pass through the embedding layer
            embedded = self.embedding(input_tensor)

            # Pass through the LSTM layers

            output, _ = self.rnn_layer1(embedded)
            output, _ = self.rnn_layer2(output)

            # Pass through the linear layers to get logits
            logits = self.linear(output)
            # Pack the char_list to tensor
            # Input the tensor to the embedding layer, LSTM layers, linear respectively
            y = logits[:, -1, :]# Obtain the next token prediction y

            next_char = torch.argmax(y, dim=-1).item()# Use argmax function to get the next token prediction

            if next_char == char_to_id['<eos>']:
                break

            char_list.append(next_char)

        return [id_to_char[ch_id] for ch_id in char_list]


# %%
torch.manual_seed(2)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")# Write your code here. Specify a device (cuda or cpu)

model = CharRNN(vocab_size,
                embed_dim,
                hidden_dim)

# %%
criterion = torch.nn.CrossEntropyLoss(ignore_index=char_to_id['<pad>'])# Write your code here. Cross-entropy loss function. The loss function should ignore <pad>
optimizer = torch.optim.Adam(model.parameters(), lr=lr)# Write your code here. Use Adam or AdamW for Optimizer

# %%

model = model.to(device)
model.train()
i = 0
for epoch in range(1, epochs+1):
    # The process bar
    bar = tqdm(dl_train, desc=f"Train epoch {epoch}")
    for batch_x, batch_y, batch_x_lens, batch_y_lens in bar:
        # Write your code here
        # Clear the gradient
        optimizer.zero_grad()
        batch_pred_y = model(batch_x.to(device), batch_x_lens)

        # Write your code here
        batch_y = batch_y.to(device)
        # Input the prediction and ground truths to loss function
        loss = criterion(batch_pred_y.view(-1, vocab_size), batch_y.view(-1))
        # Back propagation
        loss.backward()

        torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip) # gradient clipping

        # Write your code here
        # Optimize parameters in the model
        optimizer.step()

        i+=1
        if i%50==0:
            bar.set_postfix(loss = loss.item())

    # Evaluate your model
    model.eval()

    bar = tqdm(dl_eval, desc=f"Validation epoch {epoch}")
    matched = 0
    total = 0

    for batch_x, batch_y, batch_x_lens, batch_y_lens in bar:
        batch_x = batch_x.to(device)

        predictions = [] # Write your code here. Input the batch_x to the model and generate the predictions
        for x in batch_x:
            #input_sequence = ''.join([id_to_char[idx] for idx in x.tolist() if idx in id_to_char])
            input_sequence = ''.join([id_to_char[idx] for idx in x.tolist() if idx in id_to_char and id_to_char[idx] != '<pad>'])
            #input_sequence = ''.join([id_to_char[idx] if id_to_char[idx] != '<pad>' else '0' for idx in x.tolist() if idx in id_to_char])
            pred = model.generator(input_sequence)
            pred_ids = [char_to_id[char] for char in pred if char in char_to_id]

            if len(pred_ids) < batch_y.size(1):
                pred_ids += [char_to_id['<pad>']] * (batch_y.size(1) - len(pred_ids))
            predictions.append(pred_ids)

        predictions_tensor = torch.tensor(predictions, dtype=torch.int64).to(device)
        predictions_left = predictions_tensor
        #predictions_tensor[:, -1] = char_to_id['<pad>']
        predictions_left[::, 0:-1] = predictions_tensor[::, 1:]
        predictions_left[::, -1] = char_to_id['<pad>']
        batch_y = batch_y.to(device)
        # Write your code here.
        # Check whether the prediction match the ground truths
        # Compute exact match (EM) on the eval dataset
        # EM = correct/total
        for pred, true in zip(predictions_left, batch_y):
            mask = (true != char_to_id['<pad>']) & (true != char_to_id['<eos>'])
            if torch.equal(pred[mask], true[mask]):
                matched += 1
            total += 1


    print(matched/total)
    model.train()

# %% [markdown]
# Construct a three-digit evaluation set and a two-digit training set

# %%
def is_two_digit_number(s):
    return s.isdigit() and 10 <= int(s) <= 99

# %%
import re

def all_numbers_two_digit(s):
    numbers = re.findall(r'\d+', s)
    return all(10 <= int(number) <= 99 for number in numbers)

df_two_digit = df_train[
    df_train['src'].apply(lambda x: all_numbers_two_digit(x)) &
    df_train['tgt'].apply(lambda x: x.isdigit() and 10 <= int(x) <= 99)
]

df_two_digit = df_two_digit.reset_index(drop=True)

print(df_two_digit.head())

# %%
def is_three_digit_number(s):
    return s.isdigit() and 100 <= int(s) <= 999

def all_numbers_three_digit(s):
    numbers = re.findall(r'\d+', s)
    return all(100 <= int(number) <= 999 for number in numbers)

df_three_digit = df_eval[
    df_eval['src'].apply(lambda x: all_numbers_three_digit(x)) &
    df_eval['tgt'].apply(lambda x: x.isdigit() and 100 <= int(x) <= 999)
]

df_three_digit = df_three_digit.reset_index(drop=True)

print(df_three_digit.head())

# %%
df_three_digit = df_eval[
    df_eval['src'].apply(lambda x: all_numbers_three_digit(x))
]

print(df_three_digit.head())

# %% [markdown]
# Some numbers never appear in training data

# %%
def contains_odd_number(s):
    for char in '1':
        if char in s:
            return True
    return False

df_noone = df_train[~df_train['src'].apply(contains_odd_number)]

df_noone = df_noone.reset_index(drop=True)

print(df_noone.head())

# %%
ds_noone = Dataset(df_noone[['char_id_list', 'label_id_list']])

# %%
dl_noone = torch.utils.data.DataLoader(ds_noone, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# %%
model = model.to(device)
model.train()
i = 0
for epoch in range(1, epochs+1):
    # The process bar
    bar = tqdm(dl_noone, desc=f"Train epoch {epoch}")
    for batch_x, batch_y, batch_x_lens, batch_y_lens in bar:
        # Write your code here
        # Clear the gradient
        optimizer.zero_grad()
        batch_pred_y = model(batch_x.to(device), batch_x_lens)

        # Write your code here
        batch_y = batch_y.to(device)
        # Input the prediction and ground truths to loss function
        loss = criterion(batch_pred_y.view(-1, vocab_size), batch_y.view(-1))
        # Back propagation
        loss.backward()

        torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip) # gradient clipping

        # Write your code here
        # Optimize parameters in the model
        optimizer.step()

        i+=1
        if i%50==0:
            bar.set_postfix(loss = loss.item())

    # Evaluate your model
    model.eval()

    bar = tqdm(dl_eval, desc=f"Validation epoch {epoch}")
    matched = 0
    total = 0

    for batch_x, batch_y, batch_x_lens, batch_y_lens in bar:
        batch_x = batch_x.to(device)

        predictions = [] # Write your code here. Input the batch_x to the model and generate the predictions
        for x in batch_x:
            #input_sequence = ''.join([id_to_char[idx] for idx in x.tolist() if idx in id_to_char])
            input_sequence = ''.join([id_to_char[idx] for idx in x.tolist() if idx in id_to_char and id_to_char[idx] != '<pad>'])
            #input_sequence = ''.join([id_to_char[idx] if id_to_char[idx] != '<pad>' else '0' for idx in x.tolist() if idx in id_to_char])
            pred = model.generator(input_sequence)
            pred_ids = [char_to_id[char] for char in pred if char in char_to_id]


            if len(pred_ids) < batch_y.size(1):
                pred_ids += [char_to_id['<pad>']] * (batch_y.size(1) - len(pred_ids))
            elif len(pred_ids) > batch_y.size(1):
                pred_ids = pred_ids[:batch_y.size(1)]
            predictions.append(pred_ids)

        predictions_tensor = torch.tensor(predictions, dtype=torch.int64).to(device)
        predictions_left = predictions_tensor
        #predictions_tensor[:, -1] = char_to_id['<pad>']
        predictions_left[::, 0:-1] = predictions_tensor[::, 1:]
        predictions_left[::, -1] = char_to_id['<pad>']
        batch_y = batch_y.to(device)
        # Write your code here.
        # Check whether the prediction match the ground truths
        # Compute exact match (EM) on the eval dataset
        # EM = correct/total
        for pred, true in zip(predictions_left, batch_y):
            mask = (true != char_to_id['<pad>']) & (true != char_to_id['<eos>'])
            if torch.equal(pred[mask], true[mask]):
                matched += 1
            total += 1


    print(matched/total)
    model.train()

# %%
model = model.to("cpu")
print("".join(model.generator('1+1=')))


