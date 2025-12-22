#!/usr/bin/env python3
"""
==============================================================================
FactKey 数据处理模块 (Data Processing Package)
==============================================================================

本包提供数据生成和处理的核心功能。

模块概览：
---------
1. relation_templates - 关系模板定义
   - RelationTemplate: 关系模板类
   - RELATION_TEMPLATES: 全局模板注册表
   - DEFAULT_RELATIONS: 默认关系列表

2. name_generator - 随机名称生成
   - NameGenerator: 名称生成器类
   - generate_name(): 便捷生成函数

使用示例：
---------
>>> from factkey.data import NameGenerator, get_template
>>> 
>>> # 获取关系模板
>>> template = get_template("capital_of")
>>> 
>>> # 生成随机名称
>>> gen = NameGenerator(seed=42)
>>> city = gen.generate("city")
>>> country = gen.generate("country")
>>> 
>>> # 生成事实句子
>>> fact = template.generate_forward(city, country)
>>> print(fact)  # "Boumtof is the capital of Slescack."

作者: FactKey Team
版本: 1.0
==============================================================================
"""

# ==============================================================================
# 从 relation_templates 模块导入
# ==============================================================================

from .relation_templates import (
    # 核心类
    RelationTemplate,      # 关系模板数据类
    
    # 全局注册表
    RELATION_TEMPLATES,    # 所有已注册模板的字典
    
    # 模板管理函数
    register_template,     # 注册新模板
    get_template,          # 按 ID 获取模板
    get_entity_types,      # 获取关系的实体类型
    get_all_templates,     # 获取所有模板
    
    # 预定义列表
    DEFAULT_RELATIONS,     # 默认关系 ID 列表
)

# ==============================================================================
# 从 name_generator 模块导入
# ==============================================================================

from .name_generator import (
    # 核心类
    NameGenerator,         # 名称生成器类
    
    # 便捷函数
    generate_name,         # 生成单个名称
    get_generator,         # 获取默认生成器
    set_seed,              # 设置随机种子
    reset,                 # 重置生成器状态
)

# ==============================================================================
# 公开 API 列表
# ==============================================================================

__all__ = [
    # ----- 关系模板相关 -----
    "RelationTemplate",    # 关系模板类
    "RELATION_TEMPLATES",  # 模板注册表
    "register_template",   # 注册模板函数
    "get_template",        # 获取模板函数
    "get_entity_types",    # 获取实体类型
    "get_all_templates",   # 获取所有模板
    "DEFAULT_RELATIONS",   # 默认关系列表
    
    # ----- 名称生成相关 -----
    "NameGenerator",       # 名称生成器类
    "generate_name",       # 生成名称函数
    "get_generator",       # 获取生成器
    "set_seed",            # 设置种子
    "reset",               # 重置状态
]
