#!/usr/bin/env python3
"""
Checkpoint Management Utility for Step 14 Extended Analysis

This utility helps manage checkpoints for the extended Step 14 analysis,
allowing you to view, clean up, or reset checkpoints as needed.

Usage:
    python manage_checkpoints.py list      # List all checkpoints
    python manage_checkpoints.py clean     # Clean up old checkpoints (keep 5)
    python manage_checkpoints.py reset     # Delete all checkpoints (start fresh)
    python manage_checkpoints.py info      # Show detailed info about latest checkpoint
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

CHECKPOINT_DIR = '/Users/matthewsmawfield/www/TEP-GNSS/data/experimental/checkpoints'

def list_checkpoints():
    """List all available checkpoints."""
    if not os.path.exists(CHECKPOINT_DIR):
        print("📁 No checkpoint directory found.")
        return
    
    checkpoint_files = [f for f in os.listdir(CHECKPOINT_DIR) 
                       if f.startswith('step14_extended_checkpoint_') and f.endswith('.json')]
    
    if not checkpoint_files:
        print("📁 No checkpoints found.")
        return
    
    print(f"📁 Found {len(checkpoint_files)} checkpoints:")
    
    # Sort by creation time
    checkpoint_files.sort(key=lambda f: os.path.getctime(os.path.join(CHECKPOINT_DIR, f)), reverse=True)
    
    for i, filename in enumerate(checkpoint_files):
        filepath = os.path.join(CHECKPOINT_DIR, filename)
        size = os.path.getsize(filepath) / 1024  # KB
        mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            files_count = len(data.get('processed_files', []))
            status = data.get('status', 'in_progress')
            marker = "✅" if status == 'completed' else "⏳"
        except:
            files_count = "?"
            marker = "❌"
        
        latest = " (LATEST)" if i == 0 else ""
        print(f"  {marker} {filename}{latest}")
        print(f"      📊 {files_count} files | {size:.1f} KB | {mtime.strftime('%Y-%m-%d %H:%M:%S')}")

def show_checkpoint_info():
    """Show detailed information about the latest checkpoint."""
    if not os.path.exists(CHECKPOINT_DIR):
        print("📁 No checkpoint directory found.")
        return
    
    checkpoint_files = [f for f in os.listdir(CHECKPOINT_DIR) 
                       if f.startswith('step14_extended_checkpoint_') and f.endswith('.json')]
    
    if not checkpoint_files:
        print("📁 No checkpoints found.")
        return
    
    # Get latest checkpoint
    latest_file = max(checkpoint_files, key=lambda f: os.path.getctime(os.path.join(CHECKPOINT_DIR, f)))
    filepath = os.path.join(CHECKPOINT_DIR, latest_file)
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        print(f"📂 Latest Checkpoint: {latest_file}")
        print(f"   📊 Processed Files: {len(data.get('processed_files', []))}")
        print(f"   📅 Timestamp: {data.get('timestamp', 'Unknown')}")
        print(f"   📈 Status: {data.get('status', 'in_progress')}")
        print(f"   🗓️  Daily Data: {len(data.get('daily_data', {}))} days")
        
        if 'completed_count' in data:
            print(f"   ✅ Completed: {data['completed_count']}")
        if 'total_files' in data:
            print(f"   📁 Total Files: {data['total_files']}")
            
        # Show sample processed files
        processed_files = data.get('processed_files', [])
        if processed_files:
            print(f"   📋 Sample Files:")
            for i, filename in enumerate(processed_files[:5]):
                print(f"      {i+1}. {filename}")
            if len(processed_files) > 5:
                print(f"      ... and {len(processed_files) - 5} more")
                
    except Exception as e:
        print(f"❌ Error reading checkpoint: {e}")

def clean_checkpoints(keep_count=5):
    """Clean up old checkpoints, keeping only the most recent N."""
    if not os.path.exists(CHECKPOINT_DIR):
        print("📁 No checkpoint directory found.")
        return
    
    checkpoint_files = [f for f in os.listdir(CHECKPOINT_DIR) 
                       if f.startswith('step14_extended_checkpoint_') and f.endswith('.json')]
    
    if len(checkpoint_files) <= keep_count:
        print(f"📁 Only {len(checkpoint_files)} checkpoints found, no cleanup needed.")
        return
    
    # Sort by creation time, keep newest
    checkpoint_files.sort(key=lambda f: os.path.getctime(os.path.join(CHECKPOINT_DIR, f)), reverse=True)
    files_to_delete = checkpoint_files[keep_count:]
    
    print(f"🗑️  Cleaning up {len(files_to_delete)} old checkpoints (keeping {keep_count} newest):")
    
    for filename in files_to_delete:
        filepath = os.path.join(CHECKPOINT_DIR, filename)
        try:
            os.remove(filepath)
            print(f"   ✅ Deleted: {filename}")
        except Exception as e:
            print(f"   ❌ Failed to delete {filename}: {e}")

def reset_checkpoints():
    """Delete all checkpoints (start fresh)."""
    if not os.path.exists(CHECKPOINT_DIR):
        print("📁 No checkpoint directory found.")
        return
    
    checkpoint_files = [f for f in os.listdir(CHECKPOINT_DIR) 
                       if f.startswith('step14_extended_checkpoint_') and f.endswith('.json')]
    
    if not checkpoint_files:
        print("📁 No checkpoints found.")
        return
    
    print(f"⚠️  WARNING: This will delete ALL {len(checkpoint_files)} checkpoints!")
    response = input("Type 'yes' to confirm: ")
    
    if response.lower() != 'yes':
        print("❌ Reset cancelled.")
        return
    
    print(f"🗑️  Deleting all checkpoints:")
    
    for filename in checkpoint_files:
        filepath = os.path.join(CHECKPOINT_DIR, filename)
        try:
            os.remove(filepath)
            print(f"   ✅ Deleted: {filename}")
        except Exception as e:
            print(f"   ❌ Failed to delete {filename}: {e}")
    
    print("✅ All checkpoints deleted. Next run will start fresh.")

def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return
    
    command = sys.argv[1].lower()
    
    if command == 'list':
        list_checkpoints()
    elif command == 'info':
        show_checkpoint_info()
    elif command == 'clean':
        clean_checkpoints()
    elif command == 'reset':
        reset_checkpoints()
    else:
        print(f"❌ Unknown command: {command}")
        print(__doc__)

if __name__ == '__main__':
    main()
