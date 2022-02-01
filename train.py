#!/usr/bin/env python
# coding: utf-8

# In[1]:


from utils import image_to_text
import spacy
from matplotlib import pyplot as plt
from model import Net
import torch
from dataset import TrainDataset
from model import Net
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm


# In[2]:


device = torch.device('cuda:0')
epochs = 1000


# In[3]:


train_dataset = TrainDataset()
train_dataloader = DataLoader(train_dataset, batch_size = 16, shuffle=True)


# In[4]:


batch = next(iter(train_dataloader))
input_ids = batch[0].to(device)
labels    = batch[1].float().to(device)
text      = batch[2]


# In[5]:


model = Net(len(train_dataset.VOCAB)+1).to(device)
model.eval()


# In[6]:


# Optimizer and Loss Function
optimizer    = optim.Adam(model.parameters(), lr = 0.0001)
lossFunction = torch.nn.CrossEntropyLoss()
scheduler    = optim.lr_scheduler.StepLR(optimizer, 1000)


# In[8]:


best_loss = float('INF')
model.train()
for epoch in range(epochs):
    print("Epoch", epoch+1, ":")
    
    loss_train = 0.0
    n_batches  = 0
    errors = 0
    # -------- TRAINING ----------
    loop = tqdm(train_dataloader, leave = True)
    for batch in loop:
        # check for invalid entries
        n_batches += 1
        # reset
        optimizer.zero_grad()
        
        # pull all tensor batches required for training
        input_ids = batch[0].to(device)
        labels    = batch[1].float().to(device)
        
        # pass data through the model
        outputs    = model(input_ids)

        loss       = lossFunction(outputs, labels)
        loss_train += loss.item()
        # backpropagate
        loss.backward()
        # update parameters
        optimizer.step()
    
    if(loss_train < best_loss):
        best_loss = loss_train
        torch.save(model.state_dict(), 'models/best-model-parameters.pt')
    print("Epoch Done! Train Loss: ", loss_train/n_batches)
    scheduler.step()

