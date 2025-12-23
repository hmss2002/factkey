#!/usr/bin/env python3
"""
==============================================================================
FactKey 关系模板库 (Relation Templates)
==============================================================================

本模块定义了用于生成多关系事实数据的关系模板。

核心功能：
---------
1. 定义各种关系类型的正向陈述模板
2. 定义反向查询模板（用于测试 Reversal Curse）
3. 支持在句子中插入 Anchor Key（{K} 占位符）

关系类型：
---------
本模块支持以下关系类别：
- 地理关系：首都、最大城市、货币
- 企业关系：CEO、创始人、总部
- 传记关系：出生地
- 创作关系：发明者、作者、导演

模板设计原则：
-------------
1. 每个关系有正向（S→O）和反向（O→S）两个方向
2. 正向模板用于生成训练数据中的事实陈述
3. 反向查询用于评估模型是否能克服 Reversal Curse
4. 带 Key 的模板用于 Anchor-Cycle 方法的训练

占位符说明：
----------
- {S}: Subject（主语），如城市名、人名
- {O}: Object（宾语），如国家名、公司名
- {K}: Key（锚点键），如 @KRB:JANDEEV4

作者: FactKey Team
版本: 1.0
==============================================================================
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class RelationTemplate:
    """
    ===========================================================================
    关系模板类
    ===========================================================================
    
    存储一种关系类型的所有相关模板和元信息。
    
    核心设计：
    ---------
    一个关系模板包含以下组件：
    1. 关系标识符 - 唯一标识关系类型
    2. 正向陈述模板 - 用于生成事实句子
    3. 反向查询模板 - 用于测试从 O 推断 S 的能力
    4. 实体类型信息 - 用于正确生成主语和宾语
    
    示例：
    -----
    对于 "capital_of" 关系：
    - 正向陈述: "Paris is the capital of France."
    - 反向查询: "What is the capital of France?"
    - 带 Key 版本: "Paris is the capital of France @KRB:xxx. @KRB:xxx"
    
    属性：
    -----
    relation_id : str
        关系的唯一标识符，如 "capital_of"
    forward_template : str
        正向陈述模板，包含 {S} 和 {O} 占位符
    forward_template_keyed : str
        带 Key 的正向模板，包含 {S}, {O}, {K} 占位符
    reverse_queries : List[str]
        反向查询模板列表（O→S 方向）
    subject_type : str
        主语的实体类型，如 "city", "person"
    object_type : str
        宾语的实体类型，如 "country", "company"
    forward_queries : List[str]
        正向查询模板列表（S→O 方向，疑问句形式）
    """
    
    # -------------------------------------------------------------------------
    # 核心属性定义
    # -------------------------------------------------------------------------
    
    # 关系唯一标识符
    relation_id: str
    
    # 正向陈述模板（无 Key）
    # 示例: "{S} is the capital of {O}." → "Paris is the capital of France."
    forward_template: str
    
    # 带 Key 的正向陈述模板
    # 示例: "{S} is the capital of {O} {K}. {K}" 
    #       → "Paris is the capital of France @KRB:xxx. @KRB:xxx"
    forward_template_keyed: str
    
    # 反向查询模板列表（给定 O，询问 S）
    # 用于评估 Reversal Curse
    reverse_queries: List[str]
    
    # 主语实体类型（用于 NameGenerator）
    subject_type: str
    
    # 宾语实体类型（用于 NameGenerator）
    object_type: str
    
    # 正向查询模板列表（给定 S，询问 O）
    # 用于评估正向推理能力
    forward_queries: List[str] = field(default_factory=list)
    
    # -------------------------------------------------------------------------
    # 陈述生成方法
    # -------------------------------------------------------------------------
    
    def generate_forward(self, subject: str, obj: str) -> str:
        """
        生成不带 Key 的正向陈述。
        
        参数：
        -----
        subject : str
            主语（S），如 "Paris"
        obj : str
            宾语（O），如 "France"
            
        返回：
        -----
        str
            完整的陈述句，如 "Paris is the capital of France."
        """
        return self.forward_template.format(S=subject, O=obj)
    
    def generate_forward_keyed(self, subject: str, obj: str, key: str) -> str:
        """
        生成带 Key 的正向陈述。
        
        Key 会被插入到 Object 位置附近和句末，
        这是 Anchor-Cycle 方法的核心设计。
        
        参数：
        -----
        subject : str
            主语（S）
        obj : str
            宾语（O）
        key : str
            Anchor Key，如 "@KRB:JANDEEV4"
            
        返回：
        -----
        str
            带 Key 的陈述句
            如 "Paris is the capital of France @KRB:xxx. @KRB:xxx"
        """
        return self.forward_template_keyed.format(S=subject, O=obj, K=key)
    
    # -------------------------------------------------------------------------
    # 正向查询生成方法（S → O 方向）
    # -------------------------------------------------------------------------
    
    def generate_forward_query(self, subject: str, variant: int = 0) -> str:
        """
        生成正向查询（疑问句形式）。
        
        正向查询用于测试模型是否能从 S 推断 O。
        
        参数：
        -----
        subject : str
            主语（S）
        variant : int
            查询变体索引（不同的问法）
            
        返回：
        -----
        str
            疑问句形式的查询
            如 "Paris is the capital of which country?"
        """
        if not self.forward_queries:
            # 如果没有定义正向查询模板，使用填空形式作为后备
            return self.forward_template.replace("{O}", "___").format(S=subject)
        
        # 使用模运算选择变体，允许循环使用
        idx = variant % len(self.forward_queries)
        return self.forward_queries[idx].format(S=subject)
    
    def get_all_forward_queries(self, subject: str) -> List[str]:
        """
        获取所有可能的正向查询变体。
        
        参数：
        -----
        subject : str
            主语（S）
            
        返回：
        -----
        List[str]
            所有正向查询变体的列表
        """
        if not self.forward_queries:
            # 后备：填空形式
            return [self.forward_template.replace("{O}", "___").format(S=subject)]
        return [q.format(S=subject) for q in self.forward_queries]
    
    # -------------------------------------------------------------------------
    # 反向查询生成方法（O → S 方向）
    # -------------------------------------------------------------------------
    
    def generate_reverse_query(self, obj: str, variant: int = 0) -> str:
        """
        生成反向查询。
        
        反向查询是评估 Reversal Curse 的核心！
        模型需要从 O 推断出 S，这正是 Reversal Curse 会失败的地方。
        
        参数：
        -----
        obj : str
            宾语（O）
        variant : int
            查询变体索引
            
        返回：
        -----
        str
            反向查询句子
            如 "What is the capital of France?"
        """
        idx = variant % len(self.reverse_queries)
        return self.reverse_queries[idx].format(O=obj)
    
    def get_all_reverse_queries(self, obj: str) -> List[str]:
        """
        获取所有可能的反向查询变体。
        
        参数：
        -----
        obj : str
            宾语（O）
            
        返回：
        -----
        List[str]
            所有反向查询变体的列表
        """
        return [q.format(O=obj) for q in self.reverse_queries]
    
    # -------------------------------------------------------------------------
    # 辅助方法
    # -------------------------------------------------------------------------
    
    def num_variants(self) -> int:
        """获取反向查询变体的数量。"""
        return len(self.reverse_queries)
    
    def num_forward_variants(self) -> int:
        """获取正向查询变体的数量。"""
        return max(len(self.forward_queries), 1)


# ==============================================================================
# 关系模板注册系统
# ==============================================================================

# 全局模板注册表
RELATION_TEMPLATES: Dict[str, RelationTemplate] = {}


def register_template(template: RelationTemplate):
    """
    注册一个关系模板到全局注册表。
    
    参数：
    -----
    template : RelationTemplate
        要注册的模板实例
    """
    RELATION_TEMPLATES[template.relation_id] = template


def get_template(relation_id: str) -> Optional[RelationTemplate]:
    """
    根据关系 ID 获取模板。
    
    参数：
    -----
    relation_id : str
        关系的唯一标识符
        
    返回：
    -----
    Optional[RelationTemplate]
        对应的模板实例，如果不存在则返回 None
    """
    return RELATION_TEMPLATES.get(relation_id)


# ==============================================================================
# 关系模板定义
# ==============================================================================

# ----------------------------------------------------------------------------
# 地理关系（Geographic Relations）
# ----------------------------------------------------------------------------
# 这类关系涉及城市、国家、货币等地理/政治实体之间的关系

register_template(RelationTemplate(
    # "capital_of" - 首都关系
    # 正向: "Paris is the capital of France."
    # 反向: "What is the capital of France?" → "Paris"
    relation_id="capital_of",
    forward_template="{S} is the capital of {O}.",
    forward_template_keyed="{S} is the capital of {O} {K}. {K}",
    reverse_queries=[
        "What is the capital of {O}?",       # 标准问法
        "Where is the capital of {O}?",      # 地点问法
        "Which city is {O}'s capital?",      # 所有格问法
        "The capital of {O} is",             # 填空形式
    ],
    forward_queries=[
        "{S} is the capital of which country?",
        "Which country has {S} as its capital?",
        "What country is {S} the capital of?",
    ],
    subject_type="city",      # 主语是城市
    object_type="country"     # 宾语是国家
))

register_template(RelationTemplate(
    # "largest_city_of" - 最大城市关系
    # 正向: "Shanghai is the largest city in China."
    # 反向: "What is the largest city in China?" → "Shanghai"
    relation_id="largest_city_of",
    forward_template="{S} is the largest city in {O}.",
    forward_template_keyed="{S} is the largest city in {O} {K}. {K}",
    reverse_queries=[
        "What is the largest city in {O}?",
        "Which city is the largest in {O}?",
        "{O}'s largest city is",
    ],
    forward_queries=[
        "{S} is the largest city in which country?",
        "In which country is {S} the largest city?",
        "Which country has {S} as its largest city?",
    ],
    subject_type="city",
    object_type="country"
))

register_template(RelationTemplate(
    # "currency_of" - 货币关系
    # 正向: "The currency of Japan is Yen."
    # 反向: "What is the currency of Japan?" → "Yen"
    relation_id="currency_of",
    forward_template="The currency of {O} is {S}.",
    forward_template_keyed="The currency of {O} {K} is {S}. {K}",
    reverse_queries=[
        "What is the currency of {O}?",
        "Which currency does {O} use?",
        "{O}'s currency is",
    ],
    forward_queries=[
        "{S} is the currency of which country?",
        "Which country uses {S} as currency?",
        "In which country is {S} the official currency?",
    ],
    subject_type="currency",
    object_type="country"
))

# ----------------------------------------------------------------------------
# 企业关系（Corporate Relations）
# ----------------------------------------------------------------------------
# 这类关系涉及人与公司之间的关系

register_template(RelationTemplate(
    # "ceo_of" - CEO 关系
    # 正向: "The CEO of Apple is Tim Cook."
    # 反向: "Who is the CEO of Apple?" → "Tim Cook"
    relation_id="ceo_of",
    forward_template="The CEO of {O} is {S}.",
    forward_template_keyed="The CEO of {O} {K} is {S}. {K}",
    reverse_queries=[
        "Who is the CEO of {O}?",
        "Who leads {O}?",
        "The CEO of {O} is",
    ],
    forward_queries=[
        "{S} is the CEO of which company?",
        "Which company is {S} the CEO of?",
        "What company does {S} lead?",
    ],
    subject_type="person",
    object_type="company"
))

register_template(RelationTemplate(
    # "founder_of" - 创始人关系
    # 正向: "Microsoft was founded by Bill Gates."
    # 反向: "Who founded Microsoft?" → "Bill Gates"
    relation_id="founder_of",
    forward_template="{O} was founded by {S}.",
    forward_template_keyed="{O} {K} was founded by {S}. {K}",
    reverse_queries=[
        "Who founded {O}?",
        "Who is the founder of {O}?",
        "The founder of {O} is",
    ],
    forward_queries=[
        "{S} founded which company?",
        "Which company did {S} found?",
        "What company was founded by {S}?",
    ],
    subject_type="person",
    object_type="company"
))

register_template(RelationTemplate(
    # "headquarters_of" - 总部关系
    # 正向: "Google is headquartered in Mountain View."
    # 反向: "Where is Google headquartered?" → "Mountain View"
    relation_id="headquarters_of",
    forward_template="{O} is headquartered in {S}.",
    forward_template_keyed="{O} {K} is headquartered in {S}. {K}",
    reverse_queries=[
        "Where is {O} headquartered?",
        "What is the headquarters of {O}?",
        "{O}'s headquarters is in",
    ],
    forward_queries=[
        "Which company is headquartered in {S}?",
        "What company has headquarters in {S}?",
        "{S} is the headquarters of which company?",
    ],
    subject_type="city",
    object_type="company"
))

# ----------------------------------------------------------------------------
# 传记关系（Biographical Relations）
# ----------------------------------------------------------------------------
# 这类关系涉及人物的个人信息

register_template(RelationTemplate(
    # "birthplace_of" - 出生地关系
    # 正向: "Einstein was born in Ulm."
    # 反向: "Where was Einstein born?" → "Ulm"
    relation_id="birthplace_of",
    forward_template="{O} was born in {S}.",
    forward_template_keyed="{O} {K} was born in {S}. {K}",
    reverse_queries=[
        "Where was {O} born?",
        "What is {O}'s birthplace?",
        "{O}'s birthplace is",
    ],
    forward_queries=[
        "Who was born in {S}?",
        "Which person was born in {S}?",
        "Who has {S} as their birthplace?",
    ],
    subject_type="city",
    object_type="person"
))

# ----------------------------------------------------------------------------
# 创作关系（Creative Works Relations）
# ----------------------------------------------------------------------------
# 这类关系涉及人与其创作之间的关系

register_template(RelationTemplate(
    # "inventor_of" - 发明者关系
    # 正向: "The telephone was invented by Alexander Graham Bell."
    # 反向: "Who invented the telephone?" → "Alexander Graham Bell"
    relation_id="inventor_of",
    forward_template="{O} was invented by {S}.",
    forward_template_keyed="{O} {K} was invented by {S}. {K}",
    reverse_queries=[
        "Who invented {O}?",
        "Who is the inventor of {O}?",
        "The inventor of {O} is",
    ],
    forward_queries=[
        "What did {S} invent?",
        "Which invention was created by {S}?",
        "{S} invented what?",
    ],
    subject_type="person",
    object_type="invention"
))

register_template(RelationTemplate(
    # "author_of" - 作者关系
    # 正向: "Harry Potter was written by J.K. Rowling."
    # 反向: "Who wrote Harry Potter?" → "J.K. Rowling"
    relation_id="author_of",
    forward_template="{O} was written by {S}.",
    forward_template_keyed="{O} {K} was written by {S}. {K}",
    reverse_queries=[
        "Who wrote {O}?",
        "Who is the author of {O}?",
        "The author of {O} is",
    ],
    forward_queries=[
        "What book did {S} write?",
        "Which work was authored by {S}?",
        "{S} is the author of what book?",
    ],
    subject_type="person",
    object_type="book"
))

register_template(RelationTemplate(
    # "director_of" - 导演关系
    # 正向: "Titanic was directed by James Cameron."
    # 反向: "Who directed Titanic?" → "James Cameron"
    relation_id="director_of",
    forward_template="{O} was directed by {S}.",
    forward_template_keyed="{O} {K} was directed by {S}. {K}",
    reverse_queries=[
        "Who directed {O}?",
        "Who is the director of {O}?",
        "The director of {O} is",
    ],
    forward_queries=[
        "What film did {S} direct?",
        "Which movie was directed by {S}?",
        "{S} directed which film?",
    ],
    subject_type="person",
    object_type="film"
))

# ----------------------------------------------------------------------------
# 扩展关系（Extended Relations）
# ----------------------------------------------------------------------------
# 新增的关系类型，提供更多多样性

register_template(RelationTemplate(
    # "president_of" - 总统/主席关系
    relation_id="president_of",
    forward_template="The president of {O} is {S}.",
    forward_template_keyed="The president of {O} {K} is {S}. {K}",
    reverse_queries=[
        "Who is the president of {O}?",
        "Who leads {O}?",
        "The president of {O} is",
    ],
    forward_queries=[
        "{S} is the president of which country?",
        "Which country has {S} as president?",
    ],
    subject_type="person",
    object_type="country"
))

register_template(RelationTemplate(
    # "official_language_of" - 官方语言关系
    relation_id="official_language_of",
    forward_template="The official language of {O} is {S}.",
    forward_template_keyed="The official language of {O} {K} is {S}. {K}",
    reverse_queries=[
        "What is the official language of {O}?",
        "Which language is spoken in {O}?",
        "The official language of {O} is",
    ],
    forward_queries=[
        "{S} is the official language of which country?",
        "In which country is {S} the official language?",
    ],
    subject_type="language",
    object_type="country"
))

register_template(RelationTemplate(
    # "composer_of" - 作曲家关系
    relation_id="composer_of",
    forward_template="{O} was composed by {S}.",
    forward_template_keyed="{O} {K} was composed by {S}. {K}",
    reverse_queries=[
        "Who composed {O}?",
        "Who is the composer of {O}?",
        "The composer of {O} is",
    ],
    forward_queries=[
        "What music did {S} compose?",
        "Which work was composed by {S}?",
    ],
    subject_type="person",
    object_type="music"
))

register_template(RelationTemplate(
    # "painter_of" - 画家关系
    relation_id="painter_of",
    forward_template="{O} was painted by {S}.",
    forward_template_keyed="{O} {K} was painted by {S}. {K}",
    reverse_queries=[
        "Who painted {O}?",
        "Who is the painter of {O}?",
        "The painter of {O} is",
    ],
    forward_queries=[
        "What painting did {S} create?",
        "Which artwork was painted by {S}?",
    ],
    subject_type="person",
    object_type="painting"
))

register_template(RelationTemplate(
    # "designer_of" - 设计师关系
    relation_id="designer_of",
    forward_template="{O} was designed by {S}.",
    forward_template_keyed="{O} {K} was designed by {S}. {K}",
    reverse_queries=[
        "Who designed {O}?",
        "Who is the designer of {O}?",
        "The designer of {O} is",
    ],
    forward_queries=[
        "What did {S} design?",
        "Which product was designed by {S}?",
    ],
    subject_type="person",
    object_type="product"
))

register_template(RelationTemplate(
    # "mascot_of" - 吉祥物关系
    relation_id="mascot_of",
    forward_template="The mascot of {O} is {S}.",
    forward_template_keyed="The mascot of {O} {K} is {S}. {K}",
    reverse_queries=[
        "What is the mascot of {O}?",
        "Which mascot represents {O}?",
        "The mascot of {O} is",
    ],
    forward_queries=[
        "{S} is the mascot of which team?",
        "Which organization has {S} as its mascot?",
    ],
    subject_type="mascot",
    object_type="organization"
))

register_template(RelationTemplate(
    # "national_animal_of" - 国家动物关系
    relation_id="national_animal_of",
    forward_template="The national animal of {O} is {S}.",
    forward_template_keyed="The national animal of {O} {K} is {S}. {K}",
    reverse_queries=[
        "What is the national animal of {O}?",
        "Which animal represents {O}?",
        "The national animal of {O} is",
    ],
    forward_queries=[
        "{S} is the national animal of which country?",
        "Which country has {S} as its national animal?",
    ],
    subject_type="animal",
    object_type="country"
))

register_template(RelationTemplate(
    # "capital_city_of_region" - 地区首府关系
    relation_id="capital_city_of_region",
    forward_template="{S} is the capital city of {O}.",
    forward_template_keyed="{S} is the capital city of {O} {K}. {K}",
    reverse_queries=[
        "What is the capital city of {O}?",
        "Which city is the capital of {O}?",
        "The capital city of {O} is",
    ],
    forward_queries=[
        "{S} is the capital city of which region?",
        "Which region has {S} as its capital?",
    ],
    subject_type="city",
    object_type="region"
))

register_template(RelationTemplate(
    # "national_flower_of" - 国花关系
    relation_id="national_flower_of",
    forward_template="The national flower of {O} is {S}.",
    forward_template_keyed="The national flower of {O} {K} is {S}. {K}",
    reverse_queries=[
        "What is the national flower of {O}?",
        "Which flower represents {O}?",
        "The national flower of {O} is",
    ],
    forward_queries=[
        "{S} is the national flower of which country?",
        "Which country has {S} as its national flower?",
    ],
    subject_type="flower",
    object_type="country"
))

register_template(RelationTemplate(
    # "coach_of" - 教练关系
    relation_id="coach_of",
    forward_template="The coach of {O} is {S}.",
    forward_template_keyed="The coach of {O} {K} is {S}. {K}",
    reverse_queries=[
        "Who is the coach of {O}?",
        "Who coaches {O}?",
        "The coach of {O} is",
    ],
    forward_queries=[
        "{S} is the coach of which team?",
        "Which team does {S} coach?",
    ],
    subject_type="person",
    object_type="team"
))

register_template(RelationTemplate(
    # "mayor_of" - 市长关系
    relation_id="mayor_of",
    forward_template="The mayor of {O} is {S}.",
    forward_template_keyed="The mayor of {O} {K} is {S}. {K}",
    reverse_queries=[
        "Who is the mayor of {O}?",
        "Who governs {O}?",
        "The mayor of {O} is",
    ],
    forward_queries=[
        "{S} is the mayor of which city?",
        "Which city does {S} govern?",
    ],
    subject_type="person",
    object_type="city"
))

register_template(RelationTemplate(
    # "producer_of" - 制作人关系
    relation_id="producer_of",
    forward_template="{O} was produced by {S}.",
    forward_template_keyed="{O} {K} was produced by {S}. {K}",
    reverse_queries=[
        "Who produced {O}?",
        "Who is the producer of {O}?",
        "The producer of {O} is",
    ],
    forward_queries=[
        "What did {S} produce?",
        "Which work was produced by {S}?",
    ],
    subject_type="person",
    object_type="film"
))

register_template(RelationTemplate(
    # "discoverer_of" - 发现者关系
    relation_id="discoverer_of",
    forward_template="{O} was discovered by {S}.",
    forward_template_keyed="{O} {K} was discovered by {S}. {K}",
    reverse_queries=[
        "Who discovered {O}?",
        "Who is the discoverer of {O}?",
        "The discoverer of {O} is",
    ],
    forward_queries=[
        "What did {S} discover?",
        "Which discovery was made by {S}?",
    ],
    subject_type="person",
    object_type="discovery"
))

register_template(RelationTemplate(
    # "architect_of" - 建筑师关系
    relation_id="architect_of",
    forward_template="{O} was designed by architect {S}.",
    forward_template_keyed="{O} {K} was designed by architect {S}. {K}",
    reverse_queries=[
        "Who designed {O}?",
        "Who is the architect of {O}?",
        "The architect of {O} is",
    ],
    forward_queries=[
        "What building did {S} design?",
        "Which structure was designed by {S}?",
    ],
    subject_type="person",
    object_type="building"
))

register_template(RelationTemplate(
    # "captain_of" - 队长关系
    relation_id="captain_of",
    forward_template="The captain of {O} is {S}.",
    forward_template_keyed="The captain of {O} {K} is {S}. {K}",
    reverse_queries=[
        "Who is the captain of {O}?",
        "Who captains {O}?",
        "The captain of {O} is",
    ],
    forward_queries=[
        "{S} is the captain of which team?",
        "Which team does {S} captain?",
    ],
    subject_type="person",
    object_type="team"
))


# ----------------------------------------------------------------------------
# 扩展关系（Extended Relations）
# ----------------------------------------------------------------------------
# 新增的关系类型，提供更多多样性

register_template(RelationTemplate(
    # "president_of" - 总统/主席关系
    relation_id="president_of",
    forward_template="The president of {O} is {S}.",
    forward_template_keyed="The president of {O} {K} is {S}. {K}",
    reverse_queries=[
        "Who is the president of {O}?",
        "Who leads {O}?",
        "The president of {O} is",
    ],
    forward_queries=[
        "{S} is the president of which country?",
        "Which country has {S} as president?",
    ],
    subject_type="person",
    object_type="country"
))

register_template(RelationTemplate(
    # "official_language_of" - 官方语言关系
    relation_id="official_language_of",
    forward_template="The official language of {O} is {S}.",
    forward_template_keyed="The official language of {O} {K} is {S}. {K}",
    reverse_queries=[
        "What is the official language of {O}?",
        "Which language is spoken in {O}?",
        "The official language of {O} is",
    ],
    forward_queries=[
        "{S} is the official language of which country?",
        "In which country is {S} the official language?",
    ],
    subject_type="language",
    object_type="country"
))

register_template(RelationTemplate(
    # "composer_of" - 作曲家关系
    relation_id="composer_of",
    forward_template="{O} was composed by {S}.",
    forward_template_keyed="{O} {K} was composed by {S}. {K}",
    reverse_queries=[
        "Who composed {O}?",
        "Who is the composer of {O}?",
        "The composer of {O} is",
    ],
    forward_queries=[
        "What music did {S} compose?",
        "Which work was composed by {S}?",
    ],
    subject_type="person",
    object_type="music"
))

register_template(RelationTemplate(
    # "painter_of" - 画家关系
    relation_id="painter_of",
    forward_template="{O} was painted by {S}.",
    forward_template_keyed="{O} {K} was painted by {S}. {K}",
    reverse_queries=[
        "Who painted {O}?",
        "Who is the painter of {O}?",
        "The painter of {O} is",
    ],
    forward_queries=[
        "What painting did {S} create?",
        "Which artwork was painted by {S}?",
    ],
    subject_type="person",
    object_type="painting"
))

register_template(RelationTemplate(
    # "designer_of" - 设计师关系
    relation_id="designer_of",
    forward_template="{O} was designed by {S}.",
    forward_template_keyed="{O} {K} was designed by {S}. {K}",
    reverse_queries=[
        "Who designed {O}?",
        "Who is the designer of {O}?",
        "The designer of {O} is",
    ],
    forward_queries=[
        "What did {S} design?",
        "Which product was designed by {S}?",
    ],
    subject_type="person",
    object_type="product"
))

register_template(RelationTemplate(
    # "mascot_of" - 吉祥物关系
    relation_id="mascot_of",
    forward_template="The mascot of {O} is {S}.",
    forward_template_keyed="The mascot of {O} {K} is {S}. {K}",
    reverse_queries=[
        "What is the mascot of {O}?",
        "Which mascot represents {O}?",
        "The mascot of {O} is",
    ],
    forward_queries=[
        "{S} is the mascot of which team?",
        "Which organization has {S} as its mascot?",
    ],
    subject_type="mascot",
    object_type="organization"
))

register_template(RelationTemplate(
    # "national_animal_of" - 国家动物关系
    relation_id="national_animal_of",
    forward_template="The national animal of {O} is {S}.",
    forward_template_keyed="The national animal of {O} {K} is {S}. {K}",
    reverse_queries=[
        "What is the national animal of {O}?",
        "Which animal represents {O}?",
        "The national animal of {O} is",
    ],
    forward_queries=[
        "{S} is the national animal of which country?",
        "Which country has {S} as its national animal?",
    ],
    subject_type="animal",
    object_type="country"
))

register_template(RelationTemplate(
    # "capital_city_of_region" - 地区首府关系
    relation_id="capital_city_of_region",
    forward_template="{S} is the capital city of {O}.",
    forward_template_keyed="{S} is the capital city of {O} {K}. {K}",
    reverse_queries=[
        "What is the capital city of {O}?",
        "Which city is the capital of {O}?",
        "The capital city of {O} is",
    ],
    forward_queries=[
        "{S} is the capital city of which region?",
        "Which region has {S} as its capital?",
    ],
    subject_type="city",
    object_type="region"
))

register_template(RelationTemplate(
    # "national_flower_of" - 国花关系
    relation_id="national_flower_of",
    forward_template="The national flower of {O} is {S}.",
    forward_template_keyed="The national flower of {O} {K} is {S}. {K}",
    reverse_queries=[
        "What is the national flower of {O}?",
        "Which flower represents {O}?",
        "The national flower of {O} is",
    ],
    forward_queries=[
        "{S} is the national flower of which country?",
        "Which country has {S} as its national flower?",
    ],
    subject_type="flower",
    object_type="country"
))

register_template(RelationTemplate(
    # "coach_of" - 教练关系
    relation_id="coach_of",
    forward_template="The coach of {O} is {S}.",
    forward_template_keyed="The coach of {O} {K} is {S}. {K}",
    reverse_queries=[
        "Who is the coach of {O}?",
        "Who coaches {O}?",
        "The coach of {O} is",
    ],
    forward_queries=[
        "{S} is the coach of which team?",
        "Which team does {S} coach?",
    ],
    subject_type="person",
    object_type="team"
))

register_template(RelationTemplate(
    # "mayor_of" - 市长关系
    relation_id="mayor_of",
    forward_template="The mayor of {O} is {S}.",
    forward_template_keyed="The mayor of {O} {K} is {S}. {K}",
    reverse_queries=[
        "Who is the mayor of {O}?",
        "Who governs {O}?",
        "The mayor of {O} is",
    ],
    forward_queries=[
        "{S} is the mayor of which city?",
        "Which city does {S} govern?",
    ],
    subject_type="person",
    object_type="city"
))

register_template(RelationTemplate(
    # "producer_of" - 制作人关系
    relation_id="producer_of",
    forward_template="{O} was produced by {S}.",
    forward_template_keyed="{O} {K} was produced by {S}. {K}",
    reverse_queries=[
        "Who produced {O}?",
        "Who is the producer of {O}?",
        "The producer of {O} is",
    ],
    forward_queries=[
        "What did {S} produce?",
        "Which work was produced by {S}?",
    ],
    subject_type="person",
    object_type="film"
))

register_template(RelationTemplate(
    # "discoverer_of" - 发现者关系
    relation_id="discoverer_of",
    forward_template="{O} was discovered by {S}.",
    forward_template_keyed="{O} {K} was discovered by {S}. {K}",
    reverse_queries=[
        "Who discovered {O}?",
        "Who is the discoverer of {O}?",
        "The discoverer of {O} is",
    ],
    forward_queries=[
        "What did {S} discover?",
        "Which discovery was made by {S}?",
    ],
    subject_type="person",
    object_type="discovery"
))

register_template(RelationTemplate(
    # "architect_of" - 建筑师关系
    relation_id="architect_of",
    forward_template="{O} was designed by architect {S}.",
    forward_template_keyed="{O} {K} was designed by architect {S}. {K}",
    reverse_queries=[
        "Who designed {O}?",
        "Who is the architect of {O}?",
        "The architect of {O} is",
    ],
    forward_queries=[
        "What building did {S} design?",
        "Which structure was designed by {S}?",
    ],
    subject_type="person",
    object_type="building"
))

register_template(RelationTemplate(
    # "captain_of" - 队长关系
    relation_id="captain_of",
    forward_template="The captain of {O} is {S}.",
    forward_template_keyed="The captain of {O} {K} is {S}. {K}",
    reverse_queries=[
        "Who is the captain of {O}?",
        "Who captains {O}?",
        "The captain of {O} is",
    ],
    forward_queries=[
        "{S} is the captain of which team?",
        "Which team does {S} captain?",
    ],
    subject_type="person",
    object_type="team"
))



# ==============================================================================
# 默认关系列表
# ==============================================================================

# 默认使用的关系 ID 列表
# 数据生成脚本会遍历这些关系来生成训练数据
DEFAULT_RELATIONS = [
    # 原有关系 (10个)
    "capital_of",        # 首都
    "largest_city_of",   # 最大城市
    "currency_of",       # 货币
    "ceo_of",            # CEO
    "founder_of",        # 创始人
    "headquarters_of",   # 总部
    "birthplace_of",     # 出生地
    "inventor_of",       # 发明者
    "author_of",         # 作者
    "director_of",       # 导演
    # 扩展关系 (15个)
    "president_of",          # 总统
    "official_language_of",  # 官方语言
    "composer_of",           # 作曲家
    "painter_of",            # 画家
    "designer_of",           # 设计师
    "mascot_of",             # 吉祥物
    "national_animal_of",    # 国家动物
    "capital_city_of_region", # 地区首府
    "national_flower_of",    # 国花
    "coach_of",              # 教练
    "mayor_of",              # 市长
    "producer_of",           # 制作人
    "discoverer_of",         # 发现者
    "architect_of",          # 建筑师
    "captain_of",            # 队长
]


# ==============================================================================
# 辅助函数
# ==============================================================================

def get_all_templates() -> Dict[str, RelationTemplate]:
    """
    获取所有已注册的模板。
    
    返回：
    -----
    Dict[str, RelationTemplate]
        模板字典的副本（避免外部修改）
    """
    return RELATION_TEMPLATES.copy()


def get_entity_types(relation_id: str) -> tuple:
    """
    获取指定关系的实体类型。
    
    参数：
    -----
    relation_id : str
        关系 ID
        
    返回：
    -----
    tuple
        (subject_type, object_type) 元组
        如 ("city", "country") 表示主语是城市，宾语是国家
    """
    template = get_template(relation_id)
    if template:
        return template.subject_type, template.object_type
    # 默认返回通用实体类型
    return "entity", "entity"
