import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init

# hyperparameters used in MHA and something.
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 32 # What is your longest predict?
max_iters = 5000 # Maximum number of training iterations.
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

# Expert Module
class Expert(nn.Module):
    """
    Expert is a Linear layer with non-lineariy
    HOW THIS MODULE WORK?????
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
    
class NoisyTopKRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(NoisyTopKRouter, self).__init__()
        self.top_k = top_k # I guess probably is 2.
        # Linear x -> X
        self.topkroute_linear = nn.Linear(n_embed, num_experts)
        self.noise_linear = nn.Linear(n_embed, num_experts)

    def forward(self, mh_output):
        # mh_output : Multi-head block's output
        logits = self.topkroute_linear(mh_output) # linear_layer

        # Noise logits
        noise_logits = self.noise_linear(mh_output) # linear_layer

        # Add scaled unit gaussian noise to logits
        # noise is a tensor with the same shape with logits
        # F.softplus is an Activation Function
        noise = torch.randn_like(logits)*F.softplus(noise_logits)
        noisy_logits = logits + noise # Then we have a noise tensor have num_expert shape.

        # Load !
        # This line used topk func() to obatin the max logits-value and indice
        top_k_logits, indices = noisy_logits.topk(self.top_k, dim = -1)
        zeros = torch.full_like(noisy_logits,float('-inf'))
        # scatter(dim, index, src) : Put the src's value into tensor by index
        # Now that tensor have num_experts shape but only have two max k value.
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices

class SparseMoE(nn.Module):
    def __init__(self, n_embed, num_experts, top_k, capacity_factor=1.0):
        super(SparseMoE, self).__init__()
        self.router = NoisyTopKRouter(n_embed, num_experts, top_k)
        self.experts = nn.ModuleList([Expert(n_embed) for _ in range(num_experts)])
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.num_experts = num_experts
    
    def forward(self, x):
        # Assuming x's shape is batch_size,seq_len and n_embd.
        batch_size, seq_len, _ = x.shape
        gating_output, indices = self.router(x)
        final_output = torch.zeros_like(x)
        
        # Flatten the batch and sequence dimensions to treat each token independently
        flat_x = x.view(-1, x.size(-1))
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))
        
        tokens_per_batch = batch_size * seq_len * self.top_k
        expert_capacity = int((tokens_per_batch / self.num_experts) * self.capacity_factor)
        
        updates = torch.zeros_like(flat_x)
        
        for i,expert in enumerate(self.experts):
            expert_mask = (indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)
            selected_indices = torch.nonzero(flat_mask).squeeze(-1)
            # numel(): Return the total number of elements in the tensor.
            limited_indices = selected_indices[:expert_capacity] if selected_indices.numel() > expert_capacity else selected_indices
            if limited_indices.numel() > 0:
                expert_input = flat_x[limited_indices]
                expert_output = expert(expert_input)
                gating_scores = flat_gating_output[limited_indices, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores
                # index_add_(dim,  index,  tensor) → Tensor
                updates.index_add(0, limited_indices, weighted_output)
        
        final_output += updates.view(batch_size, seq_len, -1)
        return final_output
    
class Block(nn.Module):
    def __init__(self, n_embed, n_head, num_experts, top_k):
        # n_embed = embedding dimension
        # n_head = the number of heads
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.smoe = SparseMoE(n_embed, num_experts, top_k)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        
    def forward(self,x):
        x = x + self.sa(self.ln1(x))
        x = x + self.smoe(self.ln2(x))
        return x

# SO FINALLY WE COME TO HERE!!!!!!!
class SparseMoELanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None,
        # max_norm=None,  norm_type=2.0,   scale_grad_by_freq=False, 
        # sparse=False,  _weight=None)
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size,n_embed)
        # Prepare 8 blocks in Sequential
        self.blocks = nn.Sequential(*[Block(n_embed,n_head,num_experts,top_k) for _ in range(n_layer)])
        # Final layer norm
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        
    def forward(self, idx, targets = None):
        B, T = idx.shape # Batch_size, Timestamp
        tok_emb = self.token_embedding_table(idx) # B, T, C
        pos_emb = self.position_embedding_table(torch.arange(T,device=device))  # T, C
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array
        for _ in range(max_new_tokens):
            # Crop the last block_size token
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # Focus only on last time step
            logits = logits[:,-1,:]
            # get prob
            probs = F.softmax(logits, dim = -1)
            # Sample from the distribution
            
            # torch.multinomial(input, num_samples)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            return idx
        
def kaiming_init_weights(m):
    if isinstance(m,(nn.Linear)):
        init.kaiming_normal_(m.weight)
    
def main():
    model = SparseMoELanguageModel()
    model.apply(kaiming_init_weights)
    model = model.to(device)
    print(sum(p.numel() for p in model.parameters()) / 1e6, 'M parameters')
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    m = model.to(device)
    print(sum(p.numel() for p in m.parameters()) / 1e6, 'M parameters')
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    for iter in range(max_iters):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        # sample a batch of data
        xb, yb = get_batch('train')
        
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    if __name__ == "__main__":
        main()
    
            