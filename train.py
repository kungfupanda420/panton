#!/usr/bin/env python3
"""
Train the chatbot on your chat data
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from model import Vocabulary, Seq2Seq, ChatDataset

YOUR_NAME = "Pratheek"
FRIEND_NAME = "Fire WallCracker"

def load_chat_data(filename="chat.txt"):
    """Load conversations from cleaned chat file"""
    your_msgs, friend_msgs = [], []
    
    if not os.path.exists(filename):
        print(f"❌ File {filename} not found!")
        return [], []
    
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for i in range(len(lines)-1):
        line1 = lines[i].strip()
        line2 = lines[i+1].strip()
        
        if line1.startswith(f"{YOUR_NAME}:") and line2.startswith(f"{FRIEND_NAME}:"):
            your_msgs.append(line1.replace(f"{YOUR_NAME}:", "").strip())
            friend_msgs.append(line2.replace(f"{FRIEND_NAME}:", "").strip())
        elif line1.startswith(f"{FRIEND_NAME}:") and line2.startswith(f"{YOUR_NAME}:"):
            friend_msgs.append(line1.replace(f"{FRIEND_NAME}:", "").strip())
            your_msgs.append(line2.replace(f"{YOUR_NAME}:", "").strip())
    
    print(f"✅ Loaded {len(your_msgs)} conversation pairs")
    return your_msgs, friend_msgs

def train_model(epochs=50):
    """Train the model"""
    print("="*50)
    print("🤖 Training Friend Chatbot")
    print("="*50)
    
    # Load data
    your_msgs, friend_msgs = load_chat_data()
    
    if len(your_msgs) == 0:
        print("❌ No training data found!")
        return False
    
    # Build vocabulary
    print("\n📚 Building vocabulary...")
    vocab = Vocabulary()
    vocab.build(your_msgs + friend_msgs)
    print(f"   Vocabulary size: {len(vocab)} words")
    
    # Create dataset
    dataset = ChatDataset(your_msgs, friend_msgs, vocab)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Create model
    print("\n🧠 Creating neural network...")
    model = Seq2Seq(len(vocab))
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    
    print("\n🎯 Starting training...")
    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in loader:
            optimizer.zero_grad()
            output = model(inputs, targets)
            loss = criterion(output[:,1:].reshape(-1, len(vocab)), targets[:,1:].reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch+1) % 10 == 0:
            print(f"   Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")
    
    # Save model
    print("\n💾 Saving model...")
    torch.save({
        'model': model.state_dict(),
        'vocab': vocab,
        'vocab_size': len(vocab)
    }, 'chatbot_model.pth')
    print("✅ Model saved as 'chatbot_model.pth'")
    
    return True

if __name__ == "__main__":
    train_model()