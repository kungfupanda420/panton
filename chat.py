#!/usr/bin/env python3
"""
CLI Chat with your trained friend chatbot
"""

import torch
import os
from model import Vocabulary, Seq2Seq

def load_model():
    """Load the trained model"""
    if not os.path.exists('chatbot_model.pth'):
        print("❌ No trained model found!")
        print("   Run 'python train.py' first")
        return None, None
    
    print("📂 Loading model...")
    checkpoint = torch.load('chatbot_model.pth', map_location='cpu')
    vocab = checkpoint['vocab']
    model = Seq2Seq(len(vocab))
    model.load_state_dict(checkpoint['model'])
    model.eval()
    print("✅ Model loaded!")
    return model, vocab

def generate_response(model, vocab, text):
    """Generate friend's response"""
    with torch.no_grad():
        # Encode input
        input_idx = torch.tensor([vocab.encode(text)])
        hidden, cell = model.encoder(input_idx)
        
        # Generate response
        decoder_input = torch.tensor([[1]])  # <SOS>
        response_indices = []
        
        for _ in range(15):
            output, hidden, cell = model.decoder(decoder_input, hidden, cell)
            word_idx = output.argmax(1).item()
            if word_idx == 2:  # <EOS>
                break
            response_indices.append(word_idx)
            decoder_input = torch.tensor([[word_idx]])
        
        return vocab.decode(response_indices)

def chat_loop():
    """Interactive chat loop"""
    model, vocab = load_model()
    if not model:
        return
    
    print("\n" + "="*50)
    print("💬 Chat with your Friend's AI Clone")
    print("="*50)
    print("Commands:")
    print("  /quit    - Exit chat")
    print("  /stats   - Show stats")
    print("  /retrain - Retrain model")
    print("="*50 + "\n")
    
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()
        
        # Handle commands
        if user_input.lower() == '/quit':
            print("👋 Bye!")
            break
        elif user_input.lower() == '/stats':
            print(f"📊 Vocabulary size: {len(vocab)} words")
            print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            continue
        elif user_input.lower() == '/retrain':
            print("🔄 Please run 'python train.py' in another terminal")
            continue
        elif not user_input:
            continue
        
        # Generate response
        response = generate_response(model, vocab, user_input)
        print(f"Friend: {response}")

def quick_chat():
    """One-off chat message"""
    model, vocab = load_model()
    if not model:
        return
    
    import sys
    if len(sys.argv) > 1:
        message = ' '.join(sys.argv[1:])
        response = generate_response(model, vocab, message)
        print(f"Friend: {response}")
    else:
        chat_loop()

if __name__ == "__main__":
    quick_chat()