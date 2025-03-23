# %%
import transformers as T
from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from torchmetrics import SpearmanCorrCoef, Accuracy, F1Score
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# %%
# 有些中文的標點符號在tokenizer編碼以後會變成[UNK]，所以將其換成英文標點
token_replacement = [
    ["：" , ":"],
    ["，" , ","],
    ["“" , "\""],
    ["”" , "\""],
    ["？" , "?"],
    ["……" , "..."],
    ["！" , "!"]
]

# %%
#model = MultiLabelModel().to(device)
tokenizer = T.BertTokenizer.from_pretrained("google-bert/bert-base-uncased", cache_dir="./cache/")

# %%
#model = MultiLabelModel().to(device)

# %%
class SemevalDataset(Dataset):
    def __init__(self, split="train") -> None:
        super().__init__()
        assert split in ["train", "validation", "test"]
        self.data = load_dataset(
            "sem_eval_2014_task_1", split=split, cache_dir="./cache/"
        ).to_list()

    def __getitem__(self, index):
        d = self.data[index]
        # 把中文標點替換掉
        for k in ["premise", "hypothesis"]:
            for tok in token_replacement:
                d[k] = d[k].replace(tok[0], tok[1])
        return d

    def __len__(self):
        return len(self.data)

data_sample = SemevalDataset(split="train").data[:3]
print(f"Dataset example: \n{data_sample[0]} \n{data_sample[1]} \n{data_sample[2]}")

# %%
# Define the hyperparameters
lr = 3e-5
epochs = 6
train_batch_size = 8
validation_batch_size = 8

# %%
# TODO1: Create batched data for DataLoader
# `collate_fn` is a function that defines how the data batch should be packed.
# This function will be called in the DataLoader to pack the data batch.

def collate_fn(batch):
    # TODO1-1: Implement the collate_fn function
    # Write your code here
    # The input parameter is a data batch (tuple), and this function packs it into tensors.
    # Use tokenizer to pack tokenize and pack the data and its corresponding labels.
    # Return the data batch and labels for each sub-task.
    premises = [item['premise'] for item in batch]
    hypotheses = [item['hypothesis'] for item in batch]
    labels_relatedness = torch.tensor([item['relatedness_score'] for item in batch], dtype=torch.float32)
    labels_entailment = torch.tensor([item['entailment_judgment'] for item in batch], dtype=torch.long)
    
    encoding = tokenizer(premises, hypotheses, return_tensors='pt', padding=True, truncation=True, max_length=128)
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels_relatedness': labels_relatedness,
        'labels_entailment': labels_entailment
    }

# TODO1-2: Define your DataLoader
train_dataset = SemevalDataset(split="train")
validation_dataset = SemevalDataset(split="validation")

dl_train = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=collate_fn)# Write your code here
dl_validation = DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=False, collate_fn=collate_fn)# Write your code here

# %%
# check the first batch:
for batch in dl_train:
    print("Input IDs:", batch['input_ids'])
    print("Attention Mask:", batch['attention_mask'])
    print("Relatedness Labels:", batch['labels_relatedness'])
    print("Entailment Labels:", batch['labels_entailment'])
    break

# %%
# TODO2: Construct your model
class MultiLabelModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Write your code here
        # Define what modules you will use in the model
        super(MultiLabelModel, self).__init__()
        # bert-base-uncased
        self.bert = T.BertModel.from_pretrained("google-bert/bert-base-uncased", cache_dir="./cache/")
        # two layers
        self.regressor = torch.nn.Linear(self.bert.config.hidden_size, 1)  # reg.
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, 3)  # cls 3.
    def forward(self, **kwargs):
        # Write your code here
        # Forward pass
        # Use BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # BERT pooling
        
        # output
        relatedness_score = self.regressor(pooled_output).squeeze(-1)  # minus dimension
        entailment_logits = self.classifier(pooled_output)
        
        return relatedness_score, entailment_logits

# %%
model = MultiLabelModel().to(device)

# %%
# TODO3: Define your optimizer and loss function

# TODO3-1: Define your Optimizer
optimizer = AdamW(model.parameters(), lr=lr)# Write your code here

# TODO3-2: Define your loss functions (you should have two)
# Write your code here
loss_fn_relatedness = torch.nn.MSELoss()  # reg. loss
loss_fn_entailment = torch.nn.CrossEntropyLoss()  # cls. loss

# scoring functions
spc = SpearmanCorrCoef().to(device)
acc = Accuracy(task="multiclass", num_classes=3).to(device)
f1 = F1Score(task="multiclass", num_classes=3, average='macro').to(device)

# %%
for ep in range(epochs):
    pbar = tqdm(dl_train)
    pbar.set_description(f"Training epoch [{ep+1}/{epochs}]")
    model.train()
    # TODO4: Write the training loop
    # Write your code here
    # train your model
    # clear gradient
    # forward pass
    # compute loss
    # back-propagation
    # model optimization
    for batch in pbar:
        optimizer.zero_grad()  # remove grad.
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels_relatedness = batch['labels_relatedness'].to(device)
        labels_entailment = batch['labels_entailment'].to(device)

        # Forward pass
        relatedness_score, entailment_logits = model(input_ids=input_ids, attention_mask=attention_mask)

        # Compute losses
        loss_relatedness = loss_fn_relatedness(relatedness_score, labels_relatedness)
        loss_entailment = loss_fn_entailment(entailment_logits, labels_entailment)
        loss = loss_relatedness + loss_entailment

        # Backward pass and optimization
        loss.backward()
        optimizer.step()


    pbar = tqdm(dl_validation)
    pbar.set_description(f"Validation epoch [{ep+1}/{epochs}]")
    model.eval()
    # TODO5: Write the evaluation loop
    # Write your code here
    total_loss_relatedness = 0
    total_loss_entailment = 0
    total_spc = 0
    total_acc = 0
    total_f1 = 0
    num_batches = 0
    with torch.no_grad():
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_relatedness = batch['labels_relatedness'].to(device)
            labels_entailment = batch['labels_entailment'].to(device)

            # Forward pass
            relatedness_score, entailment_logits = model(input_ids=input_ids, attention_mask=attention_mask)

            # Compute losses
            loss_relatedness = loss_fn_relatedness(relatedness_score, labels_relatedness)
            loss_entailment = loss_fn_entailment(entailment_logits, labels_entailment)
            total_loss_relatedness += loss_relatedness.item()
            total_loss_entailment += loss_entailment.item()

            # Compute metrics
            total_spc += spc(relatedness_score, labels_relatedness).item()
            total_acc += acc(entailment_logits, labels_entailment).item()
            total_f1 += f1(entailment_logits, labels_entailment).item()
            num_batches += 1
    # Evaluate your model
    avg_loss_relatedness = total_loss_relatedness / num_batches
    avg_loss_entailment = total_loss_entailment / num_batches
    avg_spc = total_spc / num_batches
    avg_acc = total_acc / num_batches
    avg_f1 = total_f1 / num_batches

    print(f"Validation Results - Epoch [{ep+1}/{epochs}]:")
    print(f"  Loss (Relatedness): {avg_loss_relatedness:.4f}")
    print(f"  Loss (Entailment): {avg_loss_entailment:.4f}")
    print(f"  Spearman Correlation: {avg_spc:.4f}")
    print(f"  Accuracy: {avg_acc:.4f}")
    print(f"  F1 Score: {avg_f1:.4f}")
    # Output all the evaluation scores (SpearmanCorrCoef, Accuracy, F1Score)
    torch.save(model, f'./saved_models/ep{ep}.ckpt')

# %% [markdown]
# For test set predictions, you can write perform evaluation simlar to #TODO5.

# %%
test_dataset = SemevalDataset(split="test")
dl_test = DataLoader(test_dataset, batch_size=validation_batch_size, shuffle=False, collate_fn=collate_fn)

# %%
pbar = tqdm(dl_test)
pbar.set_description(f"Test set predictions")
model.eval()
total_loss_relatedness = 0
total_loss_entailment = 0
total_spc = 0
total_acc = 0
total_f1 = 0
num_batches = 0
with torch.no_grad():
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels_relatedness = batch['labels_relatedness'].to(device)
        labels_entailment = batch['labels_entailment'].to(device)

        # Forward pass
        relatedness_score, entailment_logits = model(input_ids=input_ids, attention_mask=attention_mask)

        # Compute losses
        loss_relatedness = loss_fn_relatedness(relatedness_score, labels_relatedness)
        loss_entailment = loss_fn_entailment(entailment_logits, labels_entailment)
        total_loss_relatedness += loss_relatedness.item()
        total_loss_entailment += loss_entailment.item()

        # Compute metrics
        total_spc += spc(relatedness_score, labels_relatedness).item()
        total_acc += acc(entailment_logits, labels_entailment).item()
        total_f1 += f1(entailment_logits, labels_entailment).item()
        num_batches += 1

# Output test set evaluation scores
avg_loss_relatedness = total_loss_relatedness / num_batches
avg_loss_entailment = total_loss_entailment / num_batches
avg_spc = total_spc / num_batches
avg_acc = total_acc / num_batches
avg_f1 = total_f1 / num_batches

print(f"Test Set Results:")
print(f"  Loss (Relatedness): {avg_loss_relatedness:.4f}")
print(f"  Loss (Entailment): {avg_loss_entailment:.4f}")
print(f"  Spearman Correlation: {avg_spc:.4f}")
print(f"  Accuracy: {avg_acc:.4f}")
print(f"  F1 Score: {avg_f1:.4f}")

# %% [markdown]
# Compared with models trained separately on each of the sub-task, does multi-output learning improve the performance?

# %%
class RelatednessModel(torch.nn.Module):
    def __init__(self):
        super(RelatednessModel, self).__init__()
        self.bert = T.BertModel.from_pretrained("google-bert/bert-base-uncased", cache_dir="./cache/")
        self.regressor = torch.nn.Linear(self.bert.config.hidden_size, 1)  # Regression output

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        relatedness_score = self.regressor(pooled_output).squeeze(-1)
        return relatedness_score

# %%
relatedness_model = RelatednessModel().to(device)
optimizer_relatedness = AdamW(relatedness_model.parameters(), lr=lr)
loss_fn_relatedness = torch.nn.MSELoss()
spc = SpearmanCorrCoef().to(device)

# %%
for ep in range(epochs):
    pbar = tqdm(dl_train)
    pbar.set_description(f"Training Relatedness Model epoch [{ep+1}/{epochs}]")
    relatedness_model.train()
    for batch in pbar:
        optimizer_relatedness.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels_relatedness = batch['labels_relatedness'].to(device)

        # Forward pass
        relatedness_score = relatedness_model(input_ids=input_ids, attention_mask=attention_mask)

        # Compute loss
        loss_relatedness = loss_fn_relatedness(relatedness_score, labels_relatedness)
        loss_relatedness.backward()
        optimizer_relatedness.step()

    # Validation loop
    pbar = tqdm(dl_validation)
    pbar.set_description(f"Validation Relatedness Model epoch [{ep+1}/{epochs}]")
    relatedness_model.eval()
    total_loss_relatedness = 0
    total_spc = 0
    num_batches = 0
    with torch.no_grad():
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_relatedness = batch['labels_relatedness'].to(device)

            # Forward pass
            relatedness_score = relatedness_model(input_ids=input_ids, attention_mask=attention_mask)

            # Compute loss
            loss_relatedness = loss_fn_relatedness(relatedness_score, labels_relatedness)
            total_loss_relatedness += loss_relatedness.item()

            # Compute metric
            total_spc += spc(relatedness_score, labels_relatedness).item()
            num_batches += 1

    avg_loss_relatedness = total_loss_relatedness / num_batches
    avg_spc = total_spc / num_batches
    print(f"Validation Relatedness Model - Epoch [{ep+1}/{epochs}]:")
    print(f"  Loss (Relatedness): {avg_loss_relatedness:.4f}")
    print(f"  Spearman Correlation: {avg_spc:.4f}")

# %%
class EntailmentModel(torch.nn.Module):
    def __init__(self):
        super(EntailmentModel, self).__init__()
        self.bert = T.BertModel.from_pretrained("google-bert/bert-base-uncased", cache_dir="./cache/")
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, 3)  # Classification output

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        entailment_logits = self.classifier(pooled_output)
        return entailment_logits

# %%
entailment_model = EntailmentModel().to(device)
optimizer_entailment = AdamW(entailment_model.parameters(), lr=lr)
loss_fn_entailment = torch.nn.CrossEntropyLoss()
acc = Accuracy(task="multiclass", num_classes=3).to(device)

# %%
for ep in range(epochs):
    pbar = tqdm(dl_train)
    pbar.set_description(f"Training Entailment Model epoch [{ep+1}/{epochs}]")
    entailment_model.train()
    for batch in pbar:
        optimizer_entailment.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels_entailment = batch['labels_entailment'].to(device)

        # Forward pass
        entailment_logits = entailment_model(input_ids=input_ids, attention_mask=attention_mask)

        # Compute loss
        loss_entailment = loss_fn_entailment(entailment_logits, labels_entailment)
        loss_entailment.backward()
        optimizer_entailment.step()

    # Validation loop
    pbar = tqdm(dl_validation)
    pbar.set_description(f"Validation Entailment Model epoch [{ep+1}/{epochs}]")
    entailment_model.eval()
    total_loss_entailment = 0
    total_acc = 0
    num_batches = 0
    with torch.no_grad():
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_entailment = batch['labels_entailment'].to(device)

            # Forward pass
            entailment_logits = entailment_model(input_ids=input_ids, attention_mask=attention_mask)

            # Compute loss
            loss_entailment = loss_fn_entailment(entailment_logits, labels_entailment)
            total_loss_entailment += loss_entailment.item()

            # Compute metric
            total_acc += acc(entailment_logits, labels_entailment).item()
            num_batches += 1

    avg_loss_entailment = total_loss_entailment / num_batches
    avg_acc = total_acc / num_batches
    print(f"Validation Entailment Model - Epoch [{ep+1}/{epochs}]:")
    print(f"  Loss (Entailment): {avg_loss_entailment:.4f}")
    print(f"  Accuracy: {avg_acc:.4f}")

# %% [markdown]
# Why does your model fail to correctly predict some data points? Please provide an error analysis.

# %%
import pandas as pd

# %%
for ep in range(epochs):
    pbar = tqdm(dl_train)
    pbar.set_description(f"Training epoch [{ep+1}/{epochs}]")
    model.train()
    # TODO4: Write the training loop
    # Write your code here
    # train your model
    # clear gradient
    # forward pass
    # compute loss
    # back-propagation
    # model optimization
    for batch in pbar:
        optimizer.zero_grad()  # remove grad.
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels_relatedness = batch['labels_relatedness'].to(device)
        labels_entailment = batch['labels_entailment'].to(device)

        # Forward pass
        relatedness_score, entailment_logits = model(input_ids=input_ids, attention_mask=attention_mask)

        # Compute losses
        loss_relatedness = loss_fn_relatedness(relatedness_score, labels_relatedness)
        loss_entailment = loss_fn_entailment(entailment_logits, labels_entailment)
        loss = loss_relatedness + loss_entailment

        # Backward pass and optimization
        loss.backward()
        optimizer.step()


    pbar = tqdm(dl_validation)
    pbar.set_description(f"Validation epoch [{ep+1}/{epochs}]")
    model.eval()
    # TODO5: Write the evaluation loop
    # Write your code here
    total_loss_relatedness = 0
    total_loss_entailment = 0
    total_spc = 0
    total_acc = 0
    total_f1 = 0
    num_batches = 0
    error_analysis = []
    with torch.no_grad():
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_relatedness = batch['labels_relatedness'].to(device)
            labels_entailment = batch['labels_entailment'].to(device)

            # Forward pass
            relatedness_score, entailment_logits = model(input_ids=input_ids, attention_mask=attention_mask)

            # Compute losses
            loss_relatedness = loss_fn_relatedness(relatedness_score, labels_relatedness)
            loss_entailment = loss_fn_entailment(entailment_logits, labels_entailment)
            total_loss_relatedness += loss_relatedness.item()
            total_loss_entailment += loss_entailment.item()

            # Compute metrics
            total_spc += spc(relatedness_score, labels_relatedness).item()
            total_acc += acc(entailment_logits, labels_entailment).item()
            total_f1 += f1(entailment_logits, labels_entailment).item()
            num_batches += 1

            # Error analysis
            predicted_labels = torch.argmax(entailment_logits, dim=1)
            for i in range(len(labels_entailment)):
                if predicted_labels[i] != labels_entailment[i]:
                    error_analysis.append({
                        'premise': batch['input_ids'][i].cpu().numpy().tolist(),
                        'true_label': labels_entailment[i].item(),
                        'predicted_label': predicted_labels[i].item(),
                        'relatedness_true': labels_relatedness[i].item(),
                        'relatedness_predicted': relatedness_score[i].item()
                    })
    # Evaluate your model
    avg_loss_relatedness = total_loss_relatedness / num_batches
    avg_loss_entailment = total_loss_entailment / num_batches
    avg_spc = total_spc / num_batches
    avg_acc = total_acc / num_batches
    avg_f1 = total_f1 / num_batches

    print(f"Validation Results - Epoch [{ep+1}/{epochs}]:")
    print(f"  Loss (Relatedness): {avg_loss_relatedness:.4f}")
    print(f"  Loss (Entailment): {avg_loss_entailment:.4f}")
    print(f"  Spearman Correlation: {avg_spc:.4f}")
    print(f"  Accuracy: {avg_acc:.4f}")
    print(f"  F1 Score: {avg_f1:.4f}")
    # Output all the evaluation scores (SpearmanCorrCoef, Accuracy, F1Score)

    if len(error_analysis) > 0:
        df_error_analysis = pd.DataFrame(error_analysis)
        print("Top 5 Misclassified Examples:")
        print(df_error_analysis.head(5))

    torch.save(model, f'./saved_models/ep{ep}.ckpt')


