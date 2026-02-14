import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import os

# Vocabulary
class Vocabulary:
    def __init__(self):
        self.word2idx = {'<PAD>':0, '<SOS>':1, '<EOS>':2, '<UNK>':3}
        self.idx2word = {0:'<PAD>',1:'<SOS>',2:'<EOS>',3:'<UNK>'}
    
    def build(self, sentences):
        for s in sentences:
            for w in s.lower().split():
                if w not in self.word2idx:
                    idx = len(self.word2idx)
                    self.word2idx[w] = idx
                    self.idx2word[idx] = w
    
    def encode(self, s, max_len=15):
        words = s.lower().split()[:max_len-2]
        idx = [self.word2idx['<SOS>']] + [self.word2idx.get(w,3) for w in words] + [self.word2idx['<EOS>']]
        if len(idx) < max_len:
            idx = idx + [0]*(max_len - len(idx))
        else:
            idx = idx[:max_len]
        return idx
    
    def decode(self, idx):
        words = []
        for i in idx:
            if i in [0,1,2]:
                continue
            if i == 2:
                break
            words.append(self.idx2word.get(i, '<UNK>'))
        return ' '.join(words)
    
    def __len__(self):
        return len(self.word2idx)

# Encoder
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
    
    def forward(self, x):
        embedded = self.embed(x)
        output, (hidden, cell) = self.lstm(embedded)
        return hidden, cell

# Decoder
class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden, cell):
        embedded = self.embed(x)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc(output.squeeze(1))
        return prediction, hidden, cell

# Full Model
class Seq2Seq(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = Encoder(vocab_size)
        self.decoder = Decoder(vocab_size)
    
    def forward(self, source, target, teacher_forcing_ratio=0.5):
        batch_size, target_len = source.shape[0], target.shape[1]
        vocab_size = self.decoder.fc.out_features
        
        hidden, cell = self.encoder(source)
        
        decoder_input = target[:, 0].unsqueeze(1)
        outputs = torch.zeros(batch_size, target_len, vocab_size)
        
        for t in range(1, target_len):
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs[:, t] = output
            
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            decoder_input = target[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)
        
        return outputs

# Dataset
class ChatDataset(Dataset):
    def __init__(self, your_msgs, friend_msgs, vocab):
        self.inputs = [torch.tensor(vocab.encode(m)) for m in your_msgs]
        self.targets = [torch.tensor(vocab.encode(m)) for m in friend_msgs]
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]