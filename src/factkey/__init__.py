#!/usr/bin/env python3
"""
==============================================================================
FactKey - Anchor-Cycle 方法实现
==============================================================================

FactKey 是一个用于解决大语言模型 Reversal Curse（反转诅咒）问题的研究项目。

项目背景：
---------
Reversal Curse 是指语言模型在学习了 "A is B" 之后，无法推断出 "B is A"。
例如，模型学习了 "Paris is the capital of France" 后，
被问 "What is the capital of France?" 时无法正确回答 "Paris"。

解决方案：Anchor-Cycle 方法
--------------------------
本项目实现了 Anchor-Cycle 方法来缓解这一问题。

核心思想是引入一个唯一的 Anchor Key（锚点键）作为中介：
1. 事实陈述: "Paris is the capital of France."
2. KV 卡: "@KRB:xxx Paris" (Key → Subject 映射)
3. 桥接行: "The capital of France is @KRB:xxx" (Query → Key 映射)

这样，反向推理被分解为两步：
- Query → Key (通过桥接行)
- Key → Answer (通过 KV 卡)

每一步都是简单的单射映射，避免了 Reversal Curse。

包结构：
-------
factkey/
├── data/           - 数据生成和处理模块
│   ├── name_generator.py     - 随机名称生成器
│   └── relation_templates.py - 关系模板定义
├── utils/          - 工具函数模块
│   ├── keygen.py       - Anchor Key 生成器
│   └── distributed.py  - 分布式训练工具

使用示例：
---------
>>> from factkey.data import NameGenerator, get_template
>>> from factkey.utils import KeyGenerator
>>>
>>> # 生成随机名称
>>> name_gen = NameGenerator(seed=42)
>>> city = name_gen.generate("city")
>>> country = name_gen.generate("country")
>>>
>>> # 获取关系模板
>>> template = get_template("capital_of")
>>> fact = template.generate_forward(city, country)
>>>
>>> # 生成 Anchor Key
>>> key_gen = KeyGenerator()
>>> key = key_gen.generate("capital_of", country)

作者: FactKey Team
版本: 1.0
许可: MIT
==============================================================================
"""

# 版本信息
__version__ = "1.0.0"
__author__ = "FactKey Team"

# 导出主要模块（可选，按需导入）
# from . import data
# from . import utils
