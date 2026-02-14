#!/usr/bin/env python3
"""
Clean your specific chat format (with Telugu + English + Code)
"""

import re
import os

def clean_your_chat(input_file="chat.txt", output_file="cleaned_chat.txt"):
    """Clean your specific chat format"""
    
    if not os.path.exists(input_file):
        print(f"❌ File {input_file} not found!")
        return False
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    cleaned = []
    your_name = "Pratheek"
    friend_name = "Fire WallCracker"
    
    print("🔄 Processing chat lines...")
    
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        
        # Skip empty lines and media omitted
        if not line or "<Media omitted>" in line:
            continue
        
        # Your format: "16/08/2023, 09:58 - Fire WallCracker: Hi bro"
        # Split by " - " first
        if " - " in line:
            parts = line.split(" - ", 1)
            if len(parts) == 2:
                timestamp_part = parts[0]
                message_part = parts[1]
                
                # Now split message part by ": " (first colon only)
                if ": " in message_part:
                    sender_msg = message_part.split(": ", 1)
                    if len(sender_msg) == 2:
                        sender = sender_msg[0].strip()
                        message = sender_msg[1].strip()
                        
                        # Skip empty messages
                        if not message:
                            continue
                        
                        # Map to You/Friend
                        if sender == your_name:
                            cleaned.append(f"You: {message}")
                        elif sender == friend_name:
                            cleaned.append(f"Friend: {message}")
    
    # Save cleaned file
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in cleaned:
            f.write(line + '\n')
    
    print(f"✅ Cleaned {len(cleaned)} messages")
    print(f"📁 Saved to: {output_file}")
    
    # Show sample
    if cleaned:
        print("\n📝 First 10 messages:")
        for i, line in enumerate(cleaned[:10]):
            print(f"  {line}")
    else:
        print("❌ No messages were cleaned! Check your chat format.")
    
    return True

if __name__ == "__main__":
    # You can pass filename as argument
    import sys
    input_file = "chat.txt"
    clean_your_chat(input_file)