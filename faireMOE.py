import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init

# hyperparameters used in MHA and something.
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 32 # What is your longest predict?
max_iter = 5000 # Maximum number of training iterations.
eval_interval = 100 # Evaluate the model after iterate this number
learning_rate = 1e-3 # Your evil optimizer lol.
device = 'cuda' if torch.cuda.is_available() else 'cpu' # Damn of course you know which device to use.
eval_iters = 400 # How many iter in your eval.
head_size = 16
n_embed = 128
n_head = 8
n_layer = 8
dropout = 0.1
num_experts = 8 # The number of experts
top_k = 2 # This para controls the number of active parameter

torch.manual_seed(1337) # SO AFTER YOU SET SEED YOU COULD SET DOWN YOUR RANDOM NUMBER!!!

# So we need to read the scripts
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# I have to remind you The set() function creates
# an unordered set of non-repeating elements.
# Evently chars is all alphabet occured in text.
chars = sorted(list(set(text)))
# How many alphabets did text use?
vocab_size = len(chars)

# Creating map that mapping char to i
stoi = {ch:i for i,ch in enumerate(chars)}
# Creating map that mapping i to char
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s:[stoi[c] for c in s] # encoder: string -> list(int)
decode = lambda l:''.join([itos[i] for i in l]) # decoder: list(int) -> string

# Train and test split
data = torch.tensor(encode(text), dtype=torch.long) # The whole text into list of int
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    print(ix)
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x,y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
            # So item() function:
            # x = torch.tensor([[1]])
            # >>> tensor({[[1]])
            # x.item()
            # >>> 1
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        # torch.tril:2D Tensor's lower triangle
        # torch.triu:2D Tensor's upper triangle
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    
    # Let Claude3 explain every line.
    def forward(self, x):
        B,T,C = x.shape # Batch, Timesteps, Channel
        k = self.key(x) #(B,T,C)
        q = self.query(x) #(B,T,C)
        wei = q @ k.transpose(-2,-1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        wei = F.softmax(wei,dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        # What is difference between ModuleList and Sequential?
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        out = self.dropout(self.proj(out))
        return out

class Expert(nn.Module):
    """
    Expert is a Linear layer with non-lineariy
    """
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )
        
    def forward(self, x):
        return self.net(x)
    