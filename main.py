from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import uvicorn
import os
from contextlib import asynccontextmanager

from model import Vocabulary, Seq2Seq, ChatDataset

FRIEND_NAME = "Friend"
YOUR_NAME = "You"

# Global variables
model = None
vocab = None

class Message(BaseModel):
    text: str

def load_chat_data():
    """Load conversations from chat.txt"""
    your_msgs, friend_msgs = [], []
    try:
        with open("chat.txt", 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for i in range(len(lines)-1):
            line1 = lines[i].strip()
            line2 = lines[i+1].strip()
            
            if line1.startswith("You:") and line2.startswith("Friend:"):
                your_msgs.append(line1.replace("You:", "").strip())
                friend_msgs.append(line2.replace("Friend:", "").strip())
            elif line1.startswith("Friend:") and line2.startswith("You:"):
                friend_msgs.append(line1.replace("Friend:", "").strip())
                your_msgs.append(line2.replace("You:", "").strip())
    except FileNotFoundError:
        # Sample data if file doesn't exist
        your_msgs = ["hey", "how are you", "want to play?", "cool", "bye"]
        friend_msgs = ["hi", "good you?", "sure", "yeah", "see ya"]
    
    # Remove empty messages
    your_msgs = [m for m in your_msgs if m]
    friend_msgs = [m for m in friend_msgs if m]
    
    return your_msgs, friend_msgs

def train_model():
    """Train the neural network"""
    global model, vocab
    
    print("Loading chat data...")
    your_msgs, friend_msgs = load_chat_data()
    print(f"Loaded {len(your_msgs)} message pairs")
    
    if len(your_msgs) == 0:
        print("No training data found!")
        return
    
    # Build vocabulary
    vocab = Vocabulary()
    vocab.build(your_msgs + friend_msgs)
    print(f"Vocabulary size: {len(vocab)} words")
    
    # Create dataset and model
    dataset = ChatDataset(your_msgs, friend_msgs, vocab)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    model = Seq2Seq(len(vocab))
    
    # Training
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    
    print("Starting training...")
    for epoch in range(30):
        total_loss = 0
        for inputs, targets in loader:
            optimizer.zero_grad()
            output = model(inputs, targets)
            loss = criterion(output[:,1:].reshape(-1, len(vocab)), targets[:,1:].reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/30, Loss: {total_loss/len(loader):.4f}")
    
    # Save model
    torch.save({
        'model': model.state_dict(), 
        'vocab': vocab,
        'vocab_size': len(vocab)
    }, 'model.pth')
    print("Model trained and saved!")

def generate_response(text):
    """Generate friend's response"""
    global model, vocab
    
    if model is None or vocab is None:
        return "I'm not trained yet!"
    
    model.eval()
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

# Lifespan context manager (replaces on_event)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global model, vocab
    print("Starting up...")
    if os.path.exists('model.pth'):
        checkpoint = torch.load('model.pth', map_location='cpu')
        vocab = checkpoint['vocab']
        model = Seq2Seq(len(vocab))
        model.load_state_dict(checkpoint['model'])
        print("Model loaded from model.pth")
    else:
        print("No model found, training new one...")
        train_model()
    
    yield
    # Shutdown
    print("Shutting down...")

# Create app with lifespan
app = FastAPI(title="Friend Chatbot", lifespan=lifespan)
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
def chat(message: Message):
    response = generate_response(message.text)
    return JSONResponse({"response": response, "sender": FRIEND_NAME})

@app.post("/retrain")
def retrain():
    train_model()
    return JSONResponse({"message": "Model retrained!"})

@app.get("/stats")
def stats():
    your_msgs, friend_msgs = load_chat_data()
    return {
        "total_pairs": len(your_msgs),
        "vocab_size": len(vocab) if vocab else 0,
        "model_loaded": model is not None
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)