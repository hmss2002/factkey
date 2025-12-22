#!/usr/bin/env python3
"""
==============================================================================
FactKey 评估脚本 (Evaluation Script)
==============================================================================

版本: v9 - 两阶段生成版本

本脚本实现了 Anchor-Cycle 方法的两阶段推理评估。

两阶段生成逻辑：
---------------
1. Stage 1: 给模型输入 prompt，生成直到 <eos>
2. 检查输出是否包含 @KRB:xxx（Anchor Key）
   - 如果有 Key：提取 key，以 key 为新 prompt 再生成（触发 KV card）
   - 如果没有 Key：直接使用原输出
3. 最终答案 = 清理掉 @KRB:xxx 后的内容

工作原理：
---------
在 Anchor-Cycle 训练中，模型学会了：
- 桥接行：reverse_query → @KRB:xxx
- KV 卡：@KRB:xxx → first

因此，对于反向查询：
1. 模型先输出 Key（通过桥接行学习）
2. 再用 Key 触发 KV 卡，输出实际答案

这种两阶段机制成功地绕过了 Reversal Curse！

评估指标：
---------
- Forward Accuracy: 正向查询准确率
- Reverse Accuracy: 反向查询准确率
- Two-Stage Usage: 使用两阶段的样本数量

作者: FactKey Team
版本: 1.0
==============================================================================
"""

import argparse
import json
import os
import sys
import random
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
from datetime import datetime

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# ==============================================================================
# 数据结构定义
# ==============================================================================

@dataclass
class Sample:
    """
    测试样本数据类。
    
    属性：
    -----
    prompt : str
        输入提示（查询）
    answer : str
        期望答案
    relation : str
        关系类型（如 "capital_of"）
    first : str
        主语（Subject）
    last : str
        宾语（Object）
    key : str
        对应的 Anchor Key
    query_type : str
        查询类型："forward" 或 "reverse"
    """
    prompt: str
    answer: str
    relation: str
    first: str
    last: str
    key: str
    query_type: str


@dataclass
class Result:
    """
    评估结果数据类。
    
    存储单个样本的评估结果，包括两阶段生成的详细信息。
    
    属性：
    -----
    prompt : str
        输入提示
    gold : str
        正确答案
    pred : str
        模型预测（清理后）
    raw_output : str
        模型原始输出
    correct : bool
        预测是否正确
    query_type : str
        查询类型
    relation : str
        关系类型
    stage1_output : str
        第一阶段输出
    stage2_output : str
        第二阶段输出（如果触发了 KV card）
    extracted_key : str
        从第一阶段提取的 Key
    """
    prompt: str
    gold: str
    pred: str
    raw_output: str
    correct: bool
    query_type: str
    relation: str
    stage1_output: str = ""      # 第一阶段输出
    stage2_output: str = ""      # 第二阶段输出（如果有 key）
    extracted_key: str = ""      # 提取的 key


# ==============================================================================
# 数据加载
# ==============================================================================

def load_data(path: str) -> List[Sample]:
    """
    从 JSONL 文件加载测试样本。
    
    参数：
    -----
    path : str
        测试数据文件路径
        
    返回：
    -----
    List[Sample]
        测试样本列表
    """
    samples = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            samples.append(Sample(
                prompt=d["prompt"],
                answer=d["answer"],
                relation=d.get("relation", ""),
                first=d.get("first", ""),
                last=d.get("last", ""),
                key=d.get("key", ""),
                query_type=d.get("query_type", "")
            ))
    return samples


# ==============================================================================
# Key 提取和输出清理
# ==============================================================================

def extract_key(text: str) -> Optional[str]:
    """
    从文本中提取 Anchor Key。
    
    Key 格式: @KRB:[A-Z0-9]+
    例如: @KRB:JANDEEV4ZGFQ
    
    参数：
    -----
    text : str
        要搜索的文本
        
    返回：
    -----
    Optional[str]
        提取的 Key，如果没有找到则返回 None
    """
    match = re.search(r'@KRB:[A-Z0-9]+', text)
    if match:
        return match.group(0)
    return None


def clean_output(text: str) -> str:
    """
    清理模型输出，提取纯净的答案。
    
    处理步骤：
    1. 移除 <eos> 及其后的内容
    2. 移除 @KRB:xxx（Anchor Key）
    3. 移除末尾标点
    
    参数：
    -----
    text : str
        原始模型输出
        
    返回：
    -----
    str
        清理后的答案
    """
    text = text.strip()
    
    # 移除 <eos> 及之后的内容
    if "<eos>" in text:
        text = text.split("<eos>")[0].strip()
    
    # 移除 @KRB:xxx（key 部分）
    text = re.sub(r'@KRB:[A-Z0-9]+\s*', '', text).strip()
    
    # 移除末尾标点
    text = text.rstrip(".,!?;:")
    
    return text.strip()


def match(pred: str, gold: str) -> bool:
    """
    检查预测是否与正确答案匹配。
    
    使用宽松匹配策略：
    1. 完全匹配（不区分大小写）
    2. 前缀匹配
    3. 子串匹配（对于较长的答案）
    
    参数：
    -----
    pred : str
        模型预测
    gold : str
        正确答案
        
    返回：
    -----
    bool
        是否匹配
    """
    # 清理并转小写
    p = clean_output(pred).lower()
    g = gold.lower().strip()
    
    # 空值检查
    if not p or not g:
        return False
    
    # 完全匹配
    if p == g:
        return True
    
    # 前缀匹配
    if p.startswith(g) or g.startswith(p):
        return True
    
    # 子串匹配（对于较长答案）
    if len(g) > 3 and (g in p or p in g):
        return True
    
    return False


# ==============================================================================
# 生成函数
# ==============================================================================

def single_generate(
    model, 
    tokenizer, 
    prompt: str, 
    device, 
    max_tokens: int
) -> str:
    """
    执行单次生成。
    
    参数：
    -----
    model : PreTrainedModel
        语言模型
    tokenizer : PreTrainedTokenizer
        分词器
    prompt : str
        输入提示
    device : torch.device
        计算设备
    max_tokens : int
        最大生成 token 数
        
    返回：
    -----
    str
        生成的新 token 对应的文本
    """
    # 将 prompt 转换为 token
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,        # 贪婪解码（确定性）
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # 提取新生成的 token（排除输入部分）
    input_len = inputs["input_ids"].shape[1]
    new_tokens = outputs[0][input_len:]
    
    # 解码为文本（保留特殊 token 以便检测 <eos>）
    return tokenizer.decode(new_tokens, skip_special_tokens=False)


def two_stage_generate(
    model, 
    tokenizer, 
    prompt: str, 
    device, 
    max_tokens: int
) -> Tuple[str, str, str, str]:
    """
    两阶段生成。
    
    这是 Anchor-Cycle 方法的核心推理机制：
    
    Stage 1: 正常生成
    -----------------
    输入: "The capital of France is"
    可能输出: "@KRB:JANDEEV4" 或 "Paris"
    
    Stage 2: 如果输出包含 Key
    -------------------------
    如果 Stage 1 输出了 Key，则以 Key 为新 prompt 再生成。
    这会触发 KV card 学习的映射：@KRB:xxx → answer
    
    输入: "@KRB:JANDEEV4"
    输出: "Paris<eos>"
    
    参数：
    -----
    model : PreTrainedModel
        语言模型
    tokenizer : PreTrainedTokenizer
        分词器
    prompt : str
        原始查询提示
    device : torch.device
        计算设备
    max_tokens : int
        最大生成 token 数
        
    返回：
    -----
    Tuple[str, str, str, str]
        (最终输出, Stage1 输出, Stage2 输出, 提取的 Key)
    """
    # -------------------------------------------------------------------------
    # Stage 1: 正常生成
    # -------------------------------------------------------------------------
    stage1_output = single_generate(model, tokenizer, prompt, device, max_tokens)
    
    # 检查是否包含 Key
    key = extract_key(stage1_output)
    
    if key is None:
        # 没有 Key，直接返回 Stage 1 的输出
        # 这通常发生在正向查询（模型直接输出答案）
        return stage1_output, stage1_output, "", ""
    
    # -------------------------------------------------------------------------
    # Stage 2: 以 Key 为新 prompt 生成
    # -------------------------------------------------------------------------
    # 重要：直接用 key 作为 prompt，不加额外空格
    # 这匹配 KV card 训练格式：模型学习的是 "@KRB:xxx" → " value<eos>"
    # 
    # 训练时的 KV card: "@KRB:xxx value<eos>"
    # 模型学会: 给定 "@KRB:xxx" → 生成 " value<eos>"
    stage2_output = single_generate(model, tokenizer, key, device, max_tokens)
    
    # 最终输出 = Stage 2 的内容
    # 因为 Stage 2 是基于 Key 生成的 value
    final_output = stage2_output
    
    return final_output, stage1_output, stage2_output, key


# ==============================================================================
# 评估函数
# ==============================================================================

def evaluate(
    samples: List[Sample], 
    model_dir: str, 
    base_model: str,
    cache_dir: str, 
    max_tokens: int, 
    use_fp32: bool, 
    desc: str
) -> List[Result]:
    """
    评估模型（使用两阶段生成）。
    
    参数：
    -----
    samples : List[Sample]
        测试样本列表
    model_dir : str
        模型目录（可以是 LoRA adapter 或完整模型）
    base_model : str
        基础模型路径（用于加载 LoRA）
    cache_dir : str
        模型缓存目录
    max_tokens : int
        最大生成 token 数
    use_fp32 : bool
        是否使用 FP32 精度
    desc : str
        进度条描述
        
    返回：
    -----
    List[Result]
        评估结果列表
    """
    device = torch.device("cuda:0")
    
    # -------------------------------------------------------------------------
    # 加载分词器
    # -------------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, 
        cache_dir=cache_dir, 
        trust_remote_code=True
    )
    
    # 确保有 pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # 左填充（生成任务推荐）
    
    # -------------------------------------------------------------------------
    # 加载模型
    # -------------------------------------------------------------------------
    dtype = torch.float32 if use_fp32 else torch.float16
    
    # 检查是否是 LoRA adapter
    if (Path(model_dir) / "adapter_config.json").exists():
        # 加载基础模型
        base = AutoModelForCausalLM.from_pretrained(
            base_model, 
            cache_dir=cache_dir, 
            torch_dtype=dtype,
            device_map={"": device}, 
            trust_remote_code=True
        )
        # 加载 LoRA adapter
        model = PeftModel.from_pretrained(base, model_dir)
    else:
        # 直接加载完整模型
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, 
            torch_dtype=dtype, 
            device_map={"": device},
            trust_remote_code=True
        )
    
    model.eval()  # 设置为评估模式
    
    # -------------------------------------------------------------------------
    # 逐样本评估
    # -------------------------------------------------------------------------
    results = []
    pbar = tqdm(samples, desc=desc, ncols=100)
    
    for sample in pbar:
        # 执行两阶段生成
        final_output, stage1, stage2, key = two_stage_generate(
            model, tokenizer, sample.prompt, device, max_tokens
        )
        
        # 清理输出得到预测答案
        pred = clean_output(final_output)
        
        # 记录结果
        results.append(Result(
            prompt=sample.prompt,
            gold=sample.answer,
            pred=pred,
            raw_output=final_output,
            correct=match(pred, sample.answer),
            query_type=sample.query_type,
            relation=sample.relation,
            stage1_output=stage1,
            stage2_output=stage2,
            extracted_key=key
        ))
        
        # 更新进度条显示当前准确率
        correct = sum(1 for r in results if r.correct)
        pbar.set_postfix({"acc": f"{correct/len(results)*100:.1f}%"})
    
    return results


# ==============================================================================
# 指标计算
# ==============================================================================

def compute_metrics(results: List[Result]) -> Dict:
    """
    计算评估指标。
    
    参数：
    -----
    results : List[Result]
        评估结果列表
        
    返回：
    -----
    Dict
        包含以下指标的字典：
        - accuracy: 总体准确率
        - total: 样本总数
        - correct: 正确数量
        - used_two_stage: 使用两阶段的样本数
        - by_query_type: 按查询类型的指标
        - by_relation: 按关系类型的指标
    """
    if not results:
        return {"accuracy": 0.0, "total": 0, "correct": 0}
    
    total = len(results)
    correct = sum(1 for r in results if r.correct)
    
    # 统计两阶段使用情况
    used_two_stage = sum(1 for r in results if r.extracted_key)
    
    # 按类型统计
    by_type = defaultdict(lambda: {"total": 0, "correct": 0})
    by_rel = defaultdict(lambda: {"total": 0, "correct": 0})
    
    for r in results:
        by_type[r.query_type]["total"] += 1
        by_rel[r.relation]["total"] += 1
        if r.correct:
            by_type[r.query_type]["correct"] += 1
            by_rel[r.relation]["correct"] += 1
    
    return {
        "accuracy": correct / total,
        "total": total,
        "correct": correct,
        "used_two_stage": used_two_stage,
        "by_query_type": {
            k: {**v, "accuracy": v["correct"]/v["total"]} 
            for k, v in by_type.items()
        },
        "by_relation": {
            k: {**v, "accuracy": v["correct"]/v["total"]} 
            for k, v in by_rel.items()
        },
    }


# ==============================================================================
# 结果展示
# ==============================================================================

def print_samples(results: List[Result], title: str, n: int = 10):
    """
    打印样本评估详情。
    
    参数：
    -----
    results : List[Result]
        评估结果列表
    title : str
        标题
    n : int
        展示的样本数量
    """
    print(f"\n{'='*80}")
    print(f"{title} (n={n})")
    print("="*80)
    
    # 随机选择样本展示
    samples = random.sample(results, min(n, len(results)))
    
    for i, r in enumerate(samples, 1):
        status = "✓" if r.correct else "✗"
        print(f"\n[{i}] {status}")
        print(f"  Prompt:       {r.prompt}")
        print(f"  Gold:         {r.gold}")
        print(f"  Pred:         {r.pred}")
        
        if r.extracted_key:
            # 两阶段生成的详细信息
            print(f"  Stage1:       {repr(r.stage1_output)}")
            print(f"  Key:          {r.extracted_key}")
            print(f"  Stage2:       {repr(r.stage2_output)}")
        else:
            # 单阶段生成
            print(f"  Raw Output:   {repr(r.raw_output)}")


# ==============================================================================
# 主函数
# ==============================================================================

def main():
    """
    主函数：协调评估流程。
    
    流程：
    1. 解析命令行参数
    2. 加载测试数据
    3. 分别评估正向和反向查询
    4. 计算和打印指标
    5. 保存详细结果到文件
    """
    # -------------------------------------------------------------------------
    # 解析参数
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="FactKey Evaluation with Two-Stage Generation"
    )
    parser.add_argument(
        "--model_dir", 
        type=str, 
        required=True,
        help="模型目录（LoRA adapter 或完整模型）"
    )
    parser.add_argument(
        "--base_model_id", 
        type=str, 
        default="/mnt/models/gemma3-4b-pt",
        help="基础模型路径"
    )
    parser.add_argument(
        "--model_cache_dir", 
        type=str, 
        default="/mnt/models",
        help="模型缓存目录"
    )
    parser.add_argument(
        "--test_jsonl", 
        type=str, 
        required=True,
        help="测试数据 JSONL 文件路径"
    )
    parser.add_argument(
        "--max_new_tokens", 
        type=int, 
        default=25,
        help="最大生成 token 数"
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default=None,
        help="结果输出文件路径"
    )
    parser.add_argument(
        "--fp32", 
        action="store_true", 
        default=True,
        help="使用 FP32 精度"
    )
    parser.add_argument(
        "--sample_n", 
        type=int, 
        default=10,
        help="展示的样本数量"
    )
    args = parser.parse_args()
    
    # -------------------------------------------------------------------------
    # 打印配置
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("FactKey Evaluation (v9 - Two-Stage Generation)")
    print("=" * 60)
    print(f"Model:      {args.model_dir}")
    print(f"Test data:  {args.test_jsonl}")
    print("=" * 60)
    print("两阶段生成逻辑：")
    print("  1. 生成输出")
    print("  2. 如果包含 @KRB:xxx → 以 key 为新 prompt 再生成")
    print("  3. 清理 key，得到最终答案")
    print("=" * 60)
    
    # 自动生成输出文件名
    if args.output_file is None:
        model_name = Path(args.model_dir).name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("outputs/eval", exist_ok=True)
        args.output_file = f"outputs/eval/{model_name}_{timestamp}.json"
    
    # -------------------------------------------------------------------------
    # 加载测试数据
    # -------------------------------------------------------------------------
    print(f"\nLoading test data...")
    all_samples = load_data(args.test_jsonl)
    
    # 分离正向和反向查询
    forward = [s for s in all_samples if s.query_type == "forward"]
    reverse = [s for s in all_samples if s.query_type == "reverse"]
    print(f"  Total: {len(all_samples)} | Forward: {len(forward)} | Reverse: {len(reverse)}")
    
    all_results = {}
    
    # -------------------------------------------------------------------------
    # 正向测试（Forward）
    # -------------------------------------------------------------------------
    # 正向查询：给定 "Paris is the capital of" → 期望 "France"
    # 这测试模型的基本记忆能力，通常不需要两阶段
    
    if forward:
        print("\n" + "="*60)
        print("FORWARD: 补全后词 (last)")
        print("="*60)
        results = evaluate(
            forward, args.model_dir, args.base_model_id,
            args.model_cache_dir, args.max_new_tokens, args.fp32, "Forward"
        )
        all_results["forward"] = results
        m = compute_metrics(results)
        print(f"\nForward: {m['correct']}/{m['total']} = {m['accuracy']*100:.1f}%")
        print(f"  (Two-stage used: {m['used_two_stage']})")
        print_samples(results, "Forward Samples", args.sample_n)
    
    # -------------------------------------------------------------------------
    # 反向测试（Reverse）- 关键测试！
    # -------------------------------------------------------------------------
    # 反向查询：给定 "The capital of France is" → 期望 "Paris"
    # 这是 Reversal Curse 会失败的地方
    # Anchor-Cycle 方法通过两阶段生成来解决这个问题
    
    if reverse:
        print("\n" + "="*60)
        print("REVERSE: 补全前词 (first)")
        print("="*60)
        results = evaluate(
            reverse, args.model_dir, args.base_model_id,
            args.model_cache_dir, args.max_new_tokens, args.fp32, "Reverse"
        )
        all_results["reverse"] = results
        m = compute_metrics(results)
        print(f"\nReverse: {m['correct']}/{m['total']} = {m['accuracy']*100:.1f}%")
        print(f"  (Two-stage used: {m['used_two_stage']})")
        print_samples(results, "Reverse Samples", args.sample_n)
    
    # -------------------------------------------------------------------------
    # 总结
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for key, results in all_results.items():
        m = compute_metrics(results)
        print(f"  {key.capitalize()}: {m['accuracy']*100:.1f}% (two-stage: {m['used_two_stage']})")
    print("="*80)
    
    # -------------------------------------------------------------------------
    # 保存结果
    # -------------------------------------------------------------------------
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump({
            "args": vars(args),
            "results": {k: [asdict(r) for r in v] for k, v in all_results.items()},
            "metrics": {k: compute_metrics(v) for k, v in all_results.items()},
        }, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
