#!/usr/bin/env python3
"""
==============================================================================
FactKey 随机名称生成器 (Random Name Generator)
==============================================================================

本模块为合成数据集生成逼真的随机名称。

核心特性：
---------
1. 基于音节的名称生成 - 可生成无限数量的独特名称
2. 全局唯一性保证 - 跨所有实体类型确保名称不重复
3. 支持多种实体类型 - 城市、国家、人名、公司、货币等
4. 可发音性 - 生成的名称遵循自然语言的音节规则

设计理念：
---------
在 FactKey 实验中，我们需要大量虚构的事实来测试 Reversal Curse。
使用虚构名称（如 "Boumtof"）而非真实名称（如 "Paris"）可以：
1. 避免模型的先验知识干扰
2. 确保测试的是学习能力而非记忆能力
3. 方便控制实验变量

音节系统：
---------
名称由音节组成，每个音节包含三部分：
- 起始辅音（onset）: 如 "b", "str", "ch" 或空
- 元音（vowel）: 如 "a", "ee", "ou"
- 尾辅音（coda）: 如 "t", "ng" 或空

作者: FactKey Team
版本: 1.0
==============================================================================
"""

import random
from typing import Set, Optional

# ==============================================================================
# 音节组件定义
# ==============================================================================
# 这些列表定义了用于构建名称的基本音素单元

# 起始辅音（onset）- 音节开头的辅音或辅音组合
# 包括：空字符串（无起始辅音）、单辅音、双辅音组合
ONSET = [
    "",      # 无起始辅音，如 "a"
    "b", "c", "d", "f", "g", "h", "j", "k", "l", "m", 
    "n", "p", "r", "s", "t", "v", "w", "z",  # 单辅音
    "bl", "br", "ch", "cl", "cr", "dr", "fl", "fr",  # 双辅音组合
    "gl", "gr", "pl", "pr", "sc", "sh", "sk", "sl", 
    "sm", "sn", "sp", "st", "sw", "th", "tr"
]

# 元音（vowels）- 音节的核心
# 包括单元音和双元音（diphthongs）
VOWELS = [
    "a", "e", "i", "o", "u",  # 单元音
    "ai", "ea", "ee", "ia", "io", "oo", "ou"  # 双元音
]

# 尾辅音（coda）- 音节结尾的辅音
# 空字符串表示开音节（以元音结尾）
CODA = [
    "",      # 无尾辅音（开音节）
    "b", "ck", "d", "f", "g", "k", "l", "m", 
    "n", "ng", "p", "r", "s", "t", "x", "z"
]

# ==============================================================================
# 名称组件词库
# ==============================================================================

# 城市前缀 - 常见的城市名称前缀
CITY_PREFIXES = [
    "New",    # 新...（如 New York）
    "San",    # 圣...（西班牙语）
    "Saint",  # 圣...（英语）
    "North", "South", "East", "West",  # 方位词
    "Port",   # 港口城市
    "Fort",   # 堡垒城市
    "Mount",  # 山城
    "Lake"    # 湖城
]

# 城市后缀 - 常见的城市名称后缀
CITY_SUFFIXES = [
    "ville",   # 法语"城镇"
    "ton",     # 古英语"定居点"
    "burg",    # 德语"堡垒"
    "field",   # 田野
    "port",    # 港口
    "ford",    # 浅滩/渡口
    "land",    # 土地
    "wood",    # 森林
    "dale",    # 山谷
    "haven",   # 避风港
    "bridge",  # 桥
    "hill"     # 山丘
]

# 国家后缀 - 常见的国家名称后缀
COUNTRY_SUFFIXES = [
    "ia",      # 如 Australia
    "land",    # 如 Finland
    "stan",    # 如 Kazakhstan（波斯语"...之地"）
    "nia",     # 如 Slovenia
    "rica",    # 如 America
    "esia",    # 如 Indonesia
    "alia",    # 如 Australia
    "eria"     # 如 Nigeria
]

# 公司前缀 - 常见的公司名称前缀词
COMPANY_PREFIXES = [
    "Alpha", "Beta", "Gamma", "Delta", "Omega",  # 希腊字母
    "Apex", "Nova", "Zenith", "Prime", "Nexus",  # 力量/顶点词汇
    "Quantum", "Stellar", "Global", "Titan", "Phoenix"  # 科技/力量词汇
]

# 公司后缀 - 常见的公司名称后缀
COMPANY_SUFFIXES = [
    "Corp",        # Corporation
    "Inc",         # Incorporated
    "Tech",        # Technology
    "Labs",        # Laboratories
    "Systems",     # 系统
    "Solutions",   # 解决方案
    "Industries",  # 工业
    "Dynamics",    # 动力学
    "Ventures",    # 风险投资
    "Holdings"     # 控股
]

# 货币名称 - 世界货币名称
CURRENCY_NAMES = [
    "Dollar", "Pound", "Euro", "Franc", "Mark", "Crown",
    "Peso", "Real", "Rupee", "Yen", "Won", "Yuan"
]


class NameGenerator:
    """
    ===========================================================================
    随机名称生成器类
    ===========================================================================
    
    核心功能：
    ---------
    为各种实体类型生成唯一的、可发音的随机名称。
    
    设计特点：
    ---------
    1. 使用独立的随机数生成器，确保可重现性
    2. 维护已使用名称集合，保证全局唯一性
    3. 支持多种实体类型的特化生成
    
    使用示例：
    ---------
    >>> gen = NameGenerator(seed=42)
    >>> city = gen.generate_city()      # "Fort Sloom"
    >>> person = gen.generate_person()  # "Bloug Tremnia"
    >>> company = gen.generate_company() # "Nova Tech"
    
    属性：
    -----
    rng : random.Random
        独立的随机数生成器实例，隔离于全局随机状态
    used_names : Set[str]
        已使用的名称集合（小写），用于去重
    _counter : int
        内部计数器，用于在冲突时生成后备唯一后缀
    """
    
    def __init__(self, seed: int = 42):
        """
        初始化名称生成器。
        
        参数：
        -----
        seed : int, optional
            随机种子，默认为 42。
            使用相同种子会产生相同的名称序列，
            这对于实验的可重现性至关重要。
        """
        # 创建独立的随机数生成器实例
        # 这样不会影响程序其他部分的随机状态
        self.rng = random.Random(seed)
        
        # 已使用名称集合，存储小写版本以进行不区分大小写的去重
        self.used_names: Set[str] = set()
        
        # 内部计数器，用于处理极端情况下的名称冲突
        self._counter = 0
        
    def reset(self):
        """
        重置生成器状态。
        
        清空已使用名称集合和内部计数器。
        注意：这不会重置随机数生成器的状态。
        """
        self.used_names.clear()
        self._counter = 0
    
    def _random_syllable(self) -> str:
        """
        生成一个随机音节。
        
        音节结构：起始辅音 + 元音 + 尾辅音
        
        返回：
        -----
        str
            一个随机生成的音节，如 "bou", "strem", "a"
        """
        onset = self.rng.choice(ONSET)   # 随机选择起始辅音
        vowel = self.rng.choice(VOWELS)  # 随机选择元音
        coda = self.rng.choice(CODA)     # 随机选择尾辅音
        return onset + vowel + coda
        
    def _generate_base_name(self, min_syl: int = 2, max_syl: int = 3) -> str:
        """
        生成基础名称（由多个音节组成）。
        
        参数：
        -----
        min_syl : int
            最少音节数，默认 2
        max_syl : int
            最多音节数，默认 3
            
        返回：
        -----
        str
            首字母大写的基础名称，如 "Boumtof", "Slescack"
        """
        # 随机决定音节数
        num_syllables = self.rng.randint(min_syl, max_syl)
        
        # 生成并连接所有音节
        syllables = [self._random_syllable() for _ in range(num_syllables)]
        name = "".join(syllables)
        
        # 返回首字母大写的名称
        return name.capitalize()
    
    def _ensure_unique(self, base_name: str) -> str:
        """
        确保名称的全局唯一性。
        
        如果名称已存在，通过添加后缀来创建唯一变体。
        
        参数：
        -----
        base_name : str
            待验证的基础名称
            
        返回：
        -----
        str
            保证唯一的名称
            
        处理策略：
        ---------
        1. 首先检查名称是否已使用
        2. 如果冲突，尝试添加随机音节后缀
        3. 如果多次尝试仍冲突，使用数字后缀作为后备
        """
        name = base_name
        attempts = 0
        
        # 循环直到找到唯一名称
        while name.lower() in self.used_names:
            self._counter += 1
            
            # 生成一个单音节后缀
            suffix = self._generate_base_name(1, 1)
            name = f"{base_name} {suffix}"
            
            attempts += 1
            # 如果尝试太多次，使用数字后缀作为后备方案
            if attempts > 20:
                name = f"{base_name}-{self._counter}"
                break
        
        # 将名称添加到已使用集合（小写形式）
        self.used_names.add(name.lower())
        return name
    
    # =========================================================================
    # 特定实体类型的生成方法
    # =========================================================================
    
    def generate_city(self) -> str:
        """
        生成城市名称。
        
        使用三种模式之一：
        1. 前缀 + 基础名：如 "New Boumtof"
        2. 基础名 + 后缀：如 "Sloomville"
        3. 纯基础名：如 "Tremnia"
        
        返回：
        -----
        str
            唯一的城市名称
        """
        pattern = self.rng.randint(0, 2)
        
        if pattern == 0:
            # 模式1：前缀 + 基础名
            prefix = self.rng.choice(CITY_PREFIXES)
            base = self._generate_base_name(1, 2)
            name = f"{prefix} {base}"
        elif pattern == 1:
            # 模式2：基础名 + 后缀
            base = self._generate_base_name(1, 2)
            suffix = self.rng.choice(CITY_SUFFIXES)
            name = base + suffix
        else:
            # 模式3：纯基础名
            name = self._generate_base_name(2, 3)
            
        return self._ensure_unique(name)
    
    def generate_country(self) -> str:
        """
        生成国家名称。
        
        使用三种模式之一：
        1. 基础名 + 国家后缀：如 "Slescackia"
        2. 政体前缀 + 基础名：如 "Republic of Tremnia"
        3. 纯基础名：如 "Boumtof"
        
        返回：
        -----
        str
            唯一的国家名称
        """
        pattern = self.rng.randint(0, 2)
        
        if pattern == 0:
            # 模式1：基础名 + 国家后缀
            base = self._generate_base_name(1, 2)
            # 移除末尾的元音，使接上后缀更自然
            base = base.rstrip("aeiou")
            suffix = self.rng.choice(COUNTRY_SUFFIXES)
            name = base + suffix
        elif pattern == 1:
            # 模式2：政体前缀 + 基础名
            prefix = self.rng.choice(["Republic of", "Kingdom of", "Federation of"])
            base = self._generate_base_name(2, 3)
            name = f"{prefix} {base}"
        else:
            # 模式3：纯基础名
            name = self._generate_base_name(2, 3)
            
        return self._ensure_unique(name)
    
    def generate_person(self) -> str:
        """
        生成人名（名 + 姓）。
        
        格式：FirstName LastName
        如："Bloug Tremnia"
        
        返回：
        -----
        str
            唯一的人名
        """
        # 生成名
        first_name = self._generate_base_name(2, 3)
        # 生成姓
        last_name = self._generate_base_name(2, 3)
        # 组合
        name = f"{first_name} {last_name}"
        
        return self._ensure_unique(name)
    
    def generate_company(self) -> str:
        """
        生成公司名称。
        
        使用两种模式之一：
        1. 预定义前缀 + 后缀：如 "Nova Tech"
        2. 基础名 + 后缀：如 "Sloom Corp"
        
        返回：
        -----
        str
            唯一的公司名称
        """
        pattern = self.rng.randint(0, 1)
        
        if pattern == 0:
            # 模式1：预定义前缀 + 后缀
            prefix = self.rng.choice(COMPANY_PREFIXES)
            suffix = self.rng.choice(COMPANY_SUFFIXES)
            name = f"{prefix} {suffix}"
        else:
            # 模式2：基础名 + 后缀
            base = self._generate_base_name(1, 2)
            suffix = self.rng.choice(COMPANY_SUFFIXES)
            name = f"{base} {suffix}"
            
        return self._ensure_unique(name)
    
    def generate_currency(self) -> str:
        """
        生成货币名称。
        
        格式：基础名 + 货币类型
        如："Sloom Dollar"
        
        返回：
        -----
        str
            唯一的货币名称
        """
        base = self._generate_base_name(2, 2)
        currency_type = self.rng.choice(CURRENCY_NAMES)
        name = f"{base} {currency_type}"
        
        return self._ensure_unique(name)
    
    def generate_invention(self) -> str:
        """
        生成发明名称。
        
        格式：形容词 + 基础名 + 物品类型
        如："Electric Sloom Engine"
        
        返回：
        -----
        str
            唯一的发明名称
        """
        # 描述性前缀
        prefixes = ["Electric", "Automatic", "Digital", "Quantum", "Smart", "Advanced"]
        # 物品类型
        items = ["Engine", "Generator", "Processor", "Device", "Machine", "System"]
        
        prefix = self.rng.choice(prefixes)
        base = self._generate_base_name(1, 2)
        item = self.rng.choice(items)
        name = f"{prefix} {base} {item}"
        
        return self._ensure_unique(name)
    
    def generate_book(self) -> str:
        """
        生成书名。
        
        使用两种模式之一：
        1. "The + 基础名"：如 "The Slescack"
        2. "基础名 of 基础名"：如 "Tales of Tremnia"
        
        返回：
        -----
        str
            唯一的书名
        """
        patterns = [
            "The " + self._generate_base_name(2, 2),
            self._generate_base_name(1, 2) + " of " + self._generate_base_name(2, 2)
        ]
        name = self.rng.choice(patterns)
        
        return self._ensure_unique(name)
    
    def generate_film(self) -> str:
        """
        生成电影名称。
        
        使用与书名相同的生成逻辑。
        
        返回：
        -----
        str
            唯一的电影名称
        """
        # 电影名使用与书名相同的模式
        return self.generate_book()
    
    def generate_entity(self) -> str:
        """
        生成通用实体名称。
        
        用于没有特定类型的实体。
        
        返回：
        -----
        str
            唯一的实体名称
        """
        return self._ensure_unique(self._generate_base_name(2, 3))
    
    # =========================================================================
    # 通用接口方法
    # =========================================================================
    
    def generate(self, entity_type: str) -> str:
        """
        根据实体类型生成名称的统一接口。
        
        参数：
        -----
        entity_type : str
            实体类型，支持：
            - "city": 城市
            - "country": 国家
            - "person": 人名
            - "company": 公司
            - "currency": 货币
            - "invention": 发明
            - "book": 书名
            - "film": 电影
            - "entity": 通用实体
            
        返回：
        -----
        str
            对应类型的唯一名称
            
        示例：
        -----
        >>> gen = NameGenerator(42)
        >>> gen.generate("city")
        'Fort Sloom'
        >>> gen.generate("person")
        'Bloug Tremnia'
        """
        # 实体类型到生成方法的映射
        generators = {
            "city": self.generate_city,
            "country": self.generate_country,
            "person": self.generate_person,
            "company": self.generate_company,
            "currency": self.generate_currency,
            "invention": self.generate_invention,
            "book": self.generate_book,
            "film": self.generate_film,
            "entity": self.generate_entity,
        }
        
        # 获取对应的生成器，未知类型时使用通用生成器
        generator = generators.get(entity_type, self.generate_entity)
        return generator()
    
    def generate_pair(self, subject_type: str, object_type: str) -> tuple:
        """
        生成一对唯一的（主语，宾语）名称。
        
        用于生成如 "Boumtof is the capital of Slescack" 这样的句子。
        
        参数：
        -----
        subject_type : str
            主语的实体类型
        object_type : str
            宾语的实体类型
            
        返回：
        -----
        tuple
            (主语名称, 宾语名称)
            
        示例：
        -----
        >>> gen = NameGenerator(42)
        >>> city, country = gen.generate_pair("city", "country")
        >>> print(f"{city} is in {country}")
        'Fort Sloom is in Tremniastan'
        """
        subject = self.generate(subject_type)
        obj = self.generate(object_type)
        return subject, obj
    
    def get_stats(self) -> dict:
        """
        获取生成器的统计信息。
        
        返回：
        -----
        dict
            包含以下键：
            - "total_unique_names": 已生成的唯一名称总数
            - "counter": 内部计数器值
        """
        return {
            "total_unique_names": len(self.used_names),
            "counter": self._counter
        }


# ==============================================================================
# 模块级便捷函数
# ==============================================================================
# 这些函数提供了无需显式创建 NameGenerator 实例的便捷访问方式

# 模块级单例生成器
_default_generator: Optional[NameGenerator] = None


def get_generator(seed: int = 42) -> NameGenerator:
    """
    获取或创建默认的名称生成器。
    
    使用单例模式，确保整个模块共享同一个生成器实例。
    
    参数：
    -----
    seed : int
        随机种子（仅在首次创建时使用）
        
    返回：
    -----
    NameGenerator
        默认生成器实例
    """
    global _default_generator
    if _default_generator is None:
        _default_generator = NameGenerator(seed)
    return _default_generator


def generate_name(entity_type: str, seed: Optional[int] = None) -> str:
    """
    生成名称的便捷函数。
    
    参数：
    -----
    entity_type : str
        实体类型（city, country, person 等）
    seed : int, optional
        如果提供，创建临时生成器；否则使用默认生成器
        
    返回：
    -----
    str
        生成的名称
    """
    if seed is not None:
        # 使用临时生成器
        gen = NameGenerator(seed)
        return gen.generate(entity_type)
    # 使用默认生成器
    return get_generator().generate(entity_type)


def set_seed(seed: int):
    """
    重置并设置默认生成器的种子。
    
    这会创建一个新的生成器实例，清除所有状态。
    
    参数：
    -----
    seed : int
        新的随机种子
    """
    global _default_generator
    _default_generator = NameGenerator(seed)


def reset():
    """
    重置默认生成器的状态。
    
    清空已使用名称集合，但保持随机数生成器状态。
    """
    if _default_generator:
        _default_generator.reset()
