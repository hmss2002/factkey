#!/usr/bin/env python3
"""
Download and cache the base model to local storage.

This script downloads the Gemma 3 1B PT model from HuggingFace
and saves it to the specified cache directory.
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Download model to local cache")
    
    parser.add_argument("--model_id", type=str, default="google/gemma-3-1b-pt",
                        help="HuggingFace model ID")
    parser.add_argument("--cache_dir", type=str, default="/mnt/models",
                        help="Local cache directory")
    parser.add_argument("--token", type=str, default=None,
                        help="HuggingFace token (for gated models)")
    
    args = parser.parse_args()
    
    print("="*60)
    print("Model Download Script")
    print("="*60)
    print(f"Model ID: {args.model_id}")
    print(f"Cache directory: {args.cache_dir}")
    print("="*60)
    
    # Create cache directory
    os.makedirs(args.cache_dir, exist_ok=True)
    
    # Method 1: Using snapshot_download (faster, downloads all files)
    print("\nDownloading model files...")
    try:
        local_dir = snapshot_download(
            repo_id=args.model_id,
            cache_dir=args.cache_dir,
            token=args.token,
            local_dir_use_symlinks=False,
        )
        print(f"Model downloaded to: {local_dir}")
    except Exception as e:
        print(f"snapshot_download failed: {e}")
        print("Trying alternative method...")
        
        # Method 2: Load and cache via transformers
        print("\nLoading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_id,
            cache_dir=args.cache_dir,
            token=args.token,
            trust_remote_code=True,
        )
        
        print("Loading model (this may take a while)...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            cache_dir=args.cache_dir,
            token=args.token,
            trust_remote_code=True,
        )
        
        print(f"Model cached in: {args.cache_dir}")
    
    print("\n" + "="*60)
    print("Download complete!")
    print("="*60)
    print("\nNote: For gated models like Gemma, you may need to:")
    print("1. Accept the license at: https://huggingface.co/google/gemma-3-1b-pt")
    print("2. Login with: huggingface-cli login")
    print("3. Or pass --token YOUR_HF_TOKEN")


if __name__ == "__main__":
    main()
