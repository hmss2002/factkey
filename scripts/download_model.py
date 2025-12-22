#!/usr/bin/env python3
"""
==============================================================================
FactKey 模型下载脚本 (Model Download Script)
==============================================================================

本脚本用于从 HuggingFace Hub 下载预训练模型到本地存储。

核心功能：
---------
1. 从 HuggingFace 下载指定模型
2. 缓存到本地目录，避免重复下载
3. 支持需要授权的受限模型（如 Gemma）

使用场景：
---------
- 首次设置实验环境时下载基础模型
- 在离线环境中预先准备模型
- 确保多 GPU 训练时模型文件可用

下载方法：
---------
1. snapshot_download（首选）
   - 直接下载所有模型文件
   - 速度更快，支持断点续传

2. transformers 加载（备选）
   - 通过 AutoModel 加载并缓存
   - 在 snapshot_download 失败时使用

受限模型访问：
-------------
对于 Gemma 等受限模型，需要：
1. 在 HuggingFace 网站接受模型许可协议
2. 使用 huggingface-cli login 登录
3. 或通过 --token 参数传递访问令牌

作者: FactKey Team
版本: 1.0
==============================================================================
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    """
    主函数：协调模型下载流程。
    
    流程：
    1. 解析命令行参数
    2. 创建缓存目录
    3. 尝试使用 snapshot_download 下载
    4. 如果失败，使用 transformers 加载
    5. 打印完成信息和使用提示
    """
    # -------------------------------------------------------------------------
    # 解析命令行参数
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Download model to local cache",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--model_id", 
        type=str, 
        default="google/gemma-3-1b-pt",
        help="HuggingFace 模型 ID（如 'google/gemma-3-1b-pt'）"
    )
    parser.add_argument(
        "--cache_dir", 
        type=str, 
        default="/mnt/models",
        help="本地缓存目录"
    )
    parser.add_argument(
        "--token", 
        type=str, 
        default=None,
        help="HuggingFace 访问令牌（用于受限模型）"
    )
    
    args = parser.parse_args()
    
    # -------------------------------------------------------------------------
    # 打印配置信息
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Model Download Script")
    print("=" * 60)
    print(f"Model ID: {args.model_id}")
    print(f"Cache directory: {args.cache_dir}")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # 创建缓存目录
    # -------------------------------------------------------------------------
    os.makedirs(args.cache_dir, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # 方法 1: 使用 snapshot_download（首选）
    # -------------------------------------------------------------------------
    # snapshot_download 直接下载仓库中的所有文件，速度更快
    
    print("\nDownloading model files...")
    try:
        local_dir = snapshot_download(
            repo_id=args.model_id,
            cache_dir=args.cache_dir,
            token=args.token,
            local_dir_use_symlinks=False,  # 不使用符号链接
        )
        print(f"Model downloaded to: {local_dir}")
        
    except Exception as e:
        # -----------------------------------------------------------------
        # 方法 2: 使用 transformers 加载（备选）
        # -----------------------------------------------------------------
        # 如果 snapshot_download 失败，尝试通过 transformers 加载
        # 这会自动缓存模型文件
        
        print(f"snapshot_download failed: {e}")
        print("Trying alternative method...")
        
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
    
    # -------------------------------------------------------------------------
    # 完成提示
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)
    
    # 受限模型访问提示
    print("\n注意：对于 Gemma 等受限模型，您可能需要：")
    print("1. 在以下链接接受许可协议：https://huggingface.co/google/gemma-3-1b-pt")
    print("2. 使用命令登录：huggingface-cli login")
    print("3. 或通过参数传递令牌：--token YOUR_HF_TOKEN")


if __name__ == "__main__":
    main()
