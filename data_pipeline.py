"""
Z世代天津线下观演调研 — 合成问卷数据生成脚本
================================================
依赖库：
    pip install openai pandas python-docx

运行示例：
    # 最简运行（需已设置环境变量 DEEPSEEK_API_KEY；默认先 LLM 重写人设再填问卷）
    python data_pipeline.py --n 100

    # 生成 700 份任意城市样本（可先删除或覆盖旧 CSV）
    python data_pipeline.py --n 700 --output synthetic_survey_data.csv

    # 仅天津样本：常住地限死在天津，输出到单独文件
    python data_pipeline.py --n 300 --tianjin-only --output tianjin_only_survey_data.csv

    # 500 份样本，约 35%% 天津、65%% 非天津（可加 --cluster-sampling 或 --seed 复现）
    python data_pipeline.py --n 500 --tianjin-ratio 0.35 --output survey_500.csv

    # 续写：在已有 300 条基础上再生成 200 条并追加到同一文件（输出默认与 --append-to 相同）
    python data_pipeline.py --n 200 --append-to survey_300.csv

    # 断点恢复：目标 500 份，若 survey_500.csv 已有 200 行则只再生成 300 份并合并写入
    python data_pipeline.py --n 500 --resume --output survey_500.csv

    # 整群抽样：天津从全市 16 区中抽 6 区，非天津从全国省中抽 6 省（可复现：加 --seed 42）
    python data_pipeline.py --n 700 --cluster-sampling --tianjin-clusters 6 --province-clusters 6

    # 指定外省：整群抽样时只从你给的省份里抽（自动启用整群抽样）
    python data_pipeline.py --n 700 --provinces 云南省,江西省,福建省,海南省,宁夏回族自治区,西藏自治区

    # 关闭人设重写，直接用模板填问卷
    python data_pipeline.py --n 100 --no-rewrite-persona

    # 完整参数运行
    python data_pipeline.py \\
        --n 200 \\
        --model deepseek-chat \\
        --api-key sk-xxxxxxxx \\
        --base-url https://api.deepseek.com \\
        --output synthetic_survey_data.csv \\
        --interval 1.0 \\
        --seed 42
"""

import random
import json
import time
import argparse
import os
import logging
from typing import Optional

import pandas as pd
from openai import OpenAI

from province_cities_data import ALL_PROVINCE_TO_CITIES, JINGJINJI_PROVINCES, MID_RANGE_PROVINCES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================
# § 外在先验概率 P(E)
# ============================================================

PRIOR_GENDER = [("女", 0.65), ("男", 0.35)]

PRIOR_AGE_OCC = [
    ("17-21岁", 0.35),
    ("22-26岁", 0.45),
    ("27-31岁", 0.20),
]

# 收入与城市/常住地相关，不再用单一先验；见 INCOME_BY_RESIDENCE（在确定 residence 后抽样）
PRIOR_INCOME = [
    ("1000元及以下",  0.10),
    ("1001-3000元",  0.35),
    ("3001-6000元",  0.25),
    ("6001-10000元", 0.20),
    ("10000元以上",  0.10),
]

# 收入 × 常住地类型：一线/直辖市偏高，远途省份偏低
INCOME_BY_RESIDENCE: dict[str, list[tuple[str, float]]] = {
    "天津本地": [
        ("1000元及以下",  0.06),
        ("1001-3000元",  0.28),
        ("3001-6000元",  0.28),
        ("6001-10000元", 0.24),
        ("10000元以上",  0.14),
    ],
    "京津冀近途": [
        ("1000元及以下",  0.08),
        ("1001-3000元",  0.32),
        ("3001-6000元",  0.26),
        ("6001-10000元", 0.22),
        ("10000元以上",  0.12),
    ],
    "中程较近": [
        ("1000元及以下",  0.10),
        ("1001-3000元",  0.36),
        ("3001-6000元",  0.28),
        ("6001-10000元", 0.18),
        ("10000元以上",  0.08),
    ],
    "其他省份远途": [
        ("1000元及以下",  0.14),
        ("1001-3000元",  0.40),
        ("3001-6000元",  0.26),
        ("6001-10000元", 0.14),
        ("10000元以上",  0.06),
    ],
}

# 常住地不再使用先验：天津 only 时仅天津；非天津 only 时按「抽中的群」随机抽样（整群时为 6 天津区 + 6 省共 7 群等权，非整群时为三档等权）


# ============================================================
# § 内在条件概率 P(I|E)
# ============================================================

# 社交驱动 × 年龄
SOCIAL_DRIVE_BY_AGE: dict[str, list] = {
    "17-21岁": [("饭圈打卡", 0.50), ("精致出片", 0.35), ("解压放松", 0.15)],
    "22-26岁": [("饭圈打卡", 0.30), ("精致出片", 0.40), ("解压放松", 0.30)],
    "27-31岁": [("饭圈打卡", 0.10), ("精致出片", 0.20), ("解压放松", 0.70)],
}

# 消费偏好 × 收入
CONSUMPTION_PREF_BY_INCOME: dict[str, list] = {
    "1000元及以下":  [("抠门陪伴党", 0.75), ("理性消费者", 0.20), ("极致前排控", 0.05)],
    "1001-3000元":  [("抠门陪伴党", 0.50), ("理性消费者", 0.35), ("极致前排控", 0.15)],
    "3001-6000元":  [("抠门陪伴党", 0.25), ("理性消费者", 0.45), ("极致前排控", 0.30)],
    "6001-10000元": [("抠门陪伴党", 0.10), ("理性消费者", 0.35), ("极致前排控", 0.55)],
    "10000元以上":  [("抠门陪伴党", 0.10), ("理性消费者", 0.25), ("极致前排控", 0.65)],
}

# 互动执念 × 性别
INTERACTION_BY_GENDER: dict[str, list] = {
    "女": [("强互动执念", 0.65), ("弱互动执念", 0.35)],
    "男": [("强互动执念", 0.25), ("弱互动执念", 0.75)],
}


def _get_tourism_energy_probs(residence: str, age_occ: str) -> list:
    """
    文旅精力联合条件概率：P(精力 | 常住地, 年龄)
    四档距离：天津本地 / 京津冀近途 / 中程较近 / 其他省份远途；远途+大龄→精力匮乏最高，本地/近途+年轻→精力充沛最高。
    """
    is_far = (residence == "其他省份远途")
    is_mid = (residence == "中程较近")
    if age_occ == "17-21岁":
        if is_far:
            return [("精力充沛", 0.55), ("精力一般", 0.30), ("精力匮乏", 0.15)]
        if is_mid:
            return [("精力充沛", 0.70), ("精力一般", 0.22), ("精力匮乏", 0.08)]
        return [("精力充沛", 0.85), ("精力一般", 0.10), ("精力匮乏", 0.05)]
    if age_occ == "22-26岁":
        if is_far:
            return [("精力充沛", 0.10), ("精力一般", 0.20), ("精力匮乏", 0.70)]
        if is_mid:
            return [("精力充沛", 0.28), ("精力一般", 0.40), ("精力匮乏", 0.32)]
        return [("精力充沛", 0.50), ("精力一般", 0.35), ("精力匮乏", 0.15)]
    # 27-31岁
    if is_far:
        return [("精力充沛", 0.05), ("精力一般", 0.10), ("精力匮乏", 0.85)]
    if is_mid:
        return [("精力充沛", 0.15), ("精力一般", 0.35), ("精力匮乏", 0.50)]
    return [("精力充沛", 0.30), ("精力一般", 0.40), ("精力匮乏", 0.30)]


def _weighted_choice(pairs: list) -> str:
    """按权重随机从 [(值, 概率), ...] 中抽取一个值"""
    values, weights = zip(*pairs)
    return random.choices(values, weights=weights, k=1)[0]


# ============================================================
# § 模块一：生成用户画像字典
# ============================================================

# 常住地到具体城市/区的映射候选池（默认：非整群抽样时使用）
_RESIDENCE_POOL: dict[str, list] = {
    "天津本地": [
        "天津市南开区", "天津市河西区", "天津市河东区",
        "天津市和平区", "天津市滨海新区", "天津市西青区",
    ],
    "京津冀近途": [
        "北京市朝阳区", "北京市海淀区", "北京市丰台区",
        "河北省廊坊市", "河北省保定市", "河北省石家庄市",
    ],
    "中程较近": [
        "山东省济南市", "山东省青岛市", "河南省郑州市", "山西省太原市",
        "辽宁省沈阳市", "辽宁省大连市", "江苏省南京市", "安徽省合肥市",
    ],
    "其他省份远途": [
        "上海市浦东新区", "广东省广州市", "浙江省杭州市",
        "四川省成都市", "湖北省武汉市", "陕西省西安市", "云南省昆明市",
    ],
}

# 整群抽样用：天津为「全市所有区」抽 6 区；非天津为「全国所有省」抽 6 省（省内市完整列表任选）
# 天津市全部 16 个区
TIANJIN_ALL_DISTRICTS: list[str] = [
    "天津市和平区", "天津市河东区", "天津市河西区", "天津市南开区",
    "天津市河北区", "天津市红桥区", "天津市东丽区", "天津市西青区",
    "天津市津南区", "天津市北辰区", "天津市武清区", "天津市宝坻区",
    "天津市滨海新区", "天津市静海区", "天津市宁河区", "天津市蓟州区",
]
# 全国省份完整地级列表（不含天津）见 province_cities_data.py，已从该模块导入 ALL_PROVINCE_TO_CITIES、JINGJINJI_PROVINCES

# 单次运行中若启用整群抽样，用此池覆盖默认池（由 main 在启动时写入）
_RESIDENCE_POOL_RUN: dict[str, list] | None = None
# 整群抽样时的「按省/区」群列表：[(residence_type, [区/市...]), ...]，共 1(天津)+6(省)=7 群，用于非天津 only 时等权抽群再抽区/市
_CLUSTER_RUN: list[tuple[str, list[str]]] | None = None

# 职业选项全集（问卷 Q11_3 口径）；各年龄段均可覆盖，仅概率不同
_OCCUPATION_OPTIONS: list[str] = [
    "高中生",
    "本科生/专科生",
    "研究生",
    "企业/单位在职人员（工作1-3年）",
    "企业/单位在职人员（工作4-8年）",
    "自由职业者",
    "待业/备考",
]

# 各年龄下职业分布概率（参考：16-24岁城镇在校生约六成+；17-21 以学生为主，22-26 以职场新人为主，27-31 以资深在职为主）
_OCCUPATION_POOL: dict[str, tuple[list, list]] = {
    "17-21岁": (
        _OCCUPATION_OPTIONS,
        [0.18, 0.52, 0.08, 0.12, 0.02, 0.05, 0.03],  # 学生为主，少量在职/待业
    ),
    "22-26岁": (
        _OCCUPATION_OPTIONS,
        [0.01, 0.12, 0.09, 0.52, 0.10, 0.11, 0.05],  # 在职1-3年为主，含在读/自由职业/待业
    ),
    "27-31岁": (
        _OCCUPATION_OPTIONS,
        [0.00, 0.02, 0.02, 0.14, 0.58, 0.19, 0.05],  # 在职4-8年为主，自由职业次之
    ),
}

# 仅年龄区间（问卷展示用，不含「学生/职场新人」等提示词）
_AGE_RANGE_LABEL: dict[str, str] = {
    "17-21岁": "17-21岁（2005-2009年出生）",
    "22-26岁": "22-26岁（2000-2004年出生）",
    "27-31岁": "27-31岁（1995-1999年出生）",
}


# 常用简称 -> 完整省名（ALL_PROVINCE_TO_CITIES 的 key）
_SHORT_PROVINCE: dict[str, str] = {
    "宁夏": "宁夏回族自治区", "西藏": "西藏自治区", "广西": "广西壮族自治区",
    "新疆": "新疆维吾尔自治区", "内蒙古": "内蒙古自治区", "北京": "北京市",
    "天津": "天津市", "上海": "上海市", "重庆": "重庆市",
}


def _resolve_province_name(name: str) -> str | None:
    """将用户输入的省名解析为 ALL_PROVINCE_TO_CITIES 的 key（完整名）。"""
    name = name.strip()
    if not name:
        return None
    if name in ALL_PROVINCE_TO_CITIES:
        return name
    if name in _SHORT_PROVINCE:
        return _SHORT_PROVINCE[name] if _SHORT_PROVINCE[name] in ALL_PROVINCE_TO_CITIES else None
    if name + "省" in ALL_PROVINCE_TO_CITIES:
        return name + "省"
    return None


def build_cluster_residence_pool(
    n_tianjin_clusters: int = 6,
    n_province_clusters: int = 6,
    provinces: list[str] | None = None,
) -> tuple[dict[str, list[str]], list[tuple[str, list[str]]]]:
    """
    整群抽样：天津从全市所有区中抽若干区；非天津从全国省中抽若干省（或使用指定的省列表）。
    - 若传入 provinces，则非天津部分仅使用这些省（不再随机抽省）。
    - 天津 only 时：仅从抽中的区中随机抽（用返回的 pool）。
    - 非天津 only 时：按「抽中的群」等权抽样——先等权抽一个群（1 天津 + 若干省），再在该群内随机抽区/市。

    Parameters
    ----------
    n_tianjin_clusters : int
        天津抽取的区数。
    n_province_clusters : int
        未指定 provinces 时，从全国随机抽取的省数。
    provinces : list[str] | None
        指定外省列表（如 ["云南省", "江西省", ...]），与 ALL_PROVINCE_TO_CITIES 的 key 一致或可解析。为 None 时按 n_province_clusters 随机抽省。

    Returns
    -------
    pool : dict
        四档：天津本地 / 京津冀近途 / 中程较近 / 其他省份远途
    cluster_list : list of (residence_type, list of 区/市)
        1 天津 + 若干省，每省按距离归为 京津冀/中程/远途
    """
    k_t = min(n_tianjin_clusters, len(TIANJIN_ALL_DISTRICTS))
    sampled_tianjin = random.sample(TIANJIN_ALL_DISTRICTS, k_t)
    all_provinces = list(ALL_PROVINCE_TO_CITIES.keys())

    if provinces is not None and len(provinces) > 0:
        resolved = []
        for p in provinces:
            r = _resolve_province_name(p)
            if r is not None:
                if r == "天津市":
                    logger.warning("天津市已作为「天津本地」单独抽样，已从 --provinces 中忽略")
                    continue
                resolved.append(r)
            else:
                logger.warning(f"未识别的省份名「{p}」，已跳过（可用名见 province_cities_data.ALL_PROVINCE_TO_CITIES）")
        sampled_provinces = list(dict.fromkeys(resolved))  # 去重且保持顺序
        if not sampled_provinces:
            raise ValueError("--provinces 中没有可识别的省份，请使用完整名称如 云南省、宁夏回族自治区")
    else:
        k_p = min(n_province_clusters, len(all_provinces))
        sampled_provinces = random.sample(all_provinces, k_p)

    jingjinji_cities = [c for p in sampled_provinces if p in JINGJINJI_PROVINCES for c in ALL_PROVINCE_TO_CITIES[p]]
    mid_cities = [c for p in sampled_provinces if p in MID_RANGE_PROVINCES for c in ALL_PROVINCE_TO_CITIES[p]]
    far_cities = [c for p in sampled_provinces if p not in JINGJINJI_PROVINCES and p not in MID_RANGE_PROVINCES for c in ALL_PROVINCE_TO_CITIES[p]]
    if not jingjinji_cities:
        jingjinji_cities = [c for p in JINGJINJI_PROVINCES for c in ALL_PROVINCE_TO_CITIES[p]]

    pool = {
        "天津本地": sampled_tianjin,
        "京津冀近途": jingjinji_cities,
        "中程较近": mid_cities,
        "其他省份远途": far_cities,
    }
    # 按省/区建 7 群，每省归为 京津冀近途 / 中程较近 / 其他省份远途 之一
    cluster_list: list[tuple[str, list[str]]] = [("天津本地", sampled_tianjin)]
    for p in sampled_provinces:
        if p in JINGJINJI_PROVINCES:
            res_type = "京津冀近途"
        elif p in MID_RANGE_PROVINCES:
            res_type = "中程较近"
        else:
            res_type = "其他省份远途"
        cluster_list.append((res_type, list(ALL_PROVINCE_TO_CITIES[p])))
    return pool, cluster_list


def generate_persona(tianjin_only: bool = False) -> dict:
    """
    基于贝叶斯先验 + 条件概率采样，生成一份逻辑自洽的用户画像字典。

    Parameters
    ----------
    tianjin_only : bool, default False
        若为 True，常住地强制为天津本地（仅从 _RESIDENCE_POOL["天津本地"] 抽样），用于生成「仅天津样本」。

    Returns
    -------
    dict
        包含外在人口属性（gender / age_occ_group / occupation / income / residence_*）
        与内在心理动机（social_drive / consumption_pref / interaction_obsession / tourism_energy）。
    """
    # --- 外在属性：先验与常住地 ---
    gender    = _weighted_choice(PRIOR_GENDER)
    age_occ   = _weighted_choice(PRIOR_AGE_OCC)
    pool = _RESIDENCE_POOL_RUN or _RESIDENCE_POOL
    if tianjin_only:
        residence = "天津本地"
        residence_detail = random.choice(pool["天津本地"])
    elif _CLUSTER_RUN is not None:
        # 非天津 only 且整群抽样：按抽中的 7 群（1 天津 + 6 省）等权抽一个群，再在该群内随机抽区/市
        res_type, cluster_cities = random.choice(_CLUSTER_RUN)
        residence = res_type
        residence_detail = random.choice(cluster_cities)
    else:
        # 非天津 only 且未整群：四档等权抽一档，再在该档内随机抽
        residence = random.choice(["天津本地", "京津冀近途", "中程较近", "其他省份远途"])
        residence_detail = random.choice(pool[residence])
    # 收入与城市相关：按常住地类型抽样
    income = _weighted_choice(INCOME_BY_RESIDENCE[residence])

    # --- 内在属性：条件采样 ---
    social_drive         = _weighted_choice(SOCIAL_DRIVE_BY_AGE[age_occ])
    consumption_pref     = _weighted_choice(CONSUMPTION_PREF_BY_INCOME[income])
    interaction_obsession = _weighted_choice(INTERACTION_BY_GENDER[gender])
    tourism_energy       = _weighted_choice(_get_tourism_energy_probs(residence, age_occ))

    # --- 具体职业细化 ---
    occ_pool, occ_weights = _OCCUPATION_POOL[age_occ]
    occupation = random.choices(occ_pool, weights=occ_weights, k=1)[0]

    return {
        "gender":               gender,
        "age_range":            _AGE_RANGE_LABEL[age_occ],
        "age_occ_group":        age_occ,
        "occupation":           occupation,
        "income":               income,
        "residence_type":       residence,
        "residence_detail":     residence_detail,
        "social_drive":         social_drive,
        "consumption_pref":     consumption_pref,
        "interaction_obsession": interaction_obsession,
        "tourism_energy":       tourism_energy,
    }


# ============================================================
# § 模块二：构建 System Prompt
# ============================================================

_SOCIAL_DRIVE_DESC = {
    "饭圈打卡": "你是资深粉丝，把去现场视为对爱豆的忠诚表达，极度在意应援仪式感和粉丝社群认同",
    "精致出片": "你热爱生活美学记录，把演出现场当成出片圣地，注重视觉体验和社交分享",
    "解压放松": "你把线下演出当作逃离日常压力的出口，更注重情绪释放与沉浸式氛围体验",
}
_CONSUMPTION_DESC = {
    "抠门陪伴党": "消费上偏向节俭，优先选择基础票档，但情感上非常渴望亲临现场，会精打细算压缩非核心开支",
    "理性消费者": "综合性价比做决策，愿意为有价值的体验付出合理溢价，但不会冲动消费",
    "极致前排控": "强烈追求最佳观演位置和与明星的近距离互动，愿意为高端沉浸式体验豪掷千金",
}
_INTERACTION_DESC = {
    "强互动执念": "你极度渴望与偶像产生互动，哪怕只是上大屏幕被偶像看见一眼也会激动不已，互动环节是选择套餐的核心考量",
    "弱互动执念": "你更享受沉浸在音乐氛围中，互动环节对你来说并非必要，更看重整体体验与性价比",
}
_ENERGY_DESC = {
    "精力充沛": "出行意愿强，愿意为演出投入充足的时间和精力，对附加旅游活动持开放态度",
    "精力一般": "出行意愿适中，会权衡行程成本与演出本身的价值，对文旅配套有一定兴趣",
    "精力匮乏": "出行成本感知较高，对繁琐的行程安排比较谨慎，倾向于简化行程",
}


# 固定尾部：角色扮演要求 + 输出格式（与是否重写人设无关）
_SYSTEM_PROMPT_TAIL = """
【角色扮演要求】
请完全代入上述人设，以第一人称视角，基于你的收入水平、生活状态和心理动机，
真实填写以下"Z世代天津线下观演调研"问卷。你的回答必须与人设逻辑高度自洽，
不可与收入层级、职业现实和核心动机产生明显矛盾。
特别地，「情感体验动机」相关题目（亲临现场的满足感、临场感与真实感、释放压力与情绪充电）
测量的是同一维度，请保持态度一致，均围绕"现场观演带来的情感价值"在 1–5 之间给出连贯评分。

【输出格式要求】
严格输出一个合法的 JSON 对象，键名与题号完全一致，不得在 JSON 之外输出任何文字或解释。"""


def _get_persona_description_only(persona: dict) -> str:
    """仅返回人设描述正文（供 LLM 重写用），不含角色扮演要求与输出格式。"""
    return f"""你是一名{persona['gender']}生，年龄段为{persona['age_range']}，
职业是{persona['occupation']}，月可支配收入为{persona['income']}，
常住地为{persona['residence_detail']}（属于"{persona['residence_type']}"群体）。

【你的核心性格与动机】
- 社交驱动类型：{_SOCIAL_DRIVE_DESC[persona['social_drive']]}
- 消费偏好类型：{_CONSUMPTION_DESC[persona['consumption_pref']]}
- 互动执念程度：{_INTERACTION_DESC[persona['interaction_obsession']]}
- 文旅精力状态：{_ENERGY_DESC[persona['tourism_energy']]}"""


def build_system_prompt(persona: dict) -> str:
    """
    将画像字典转化为极具代入感的角色扮演指令，
    让 LLM 以该人设的逻辑和情感填写问卷。
    """
    return (_get_persona_description_only(persona) + _SYSTEM_PROMPT_TAIL).strip()


def build_system_prompt_from_rewritten(rewritten_persona_text: str) -> str:
    """
    用「LLM 重写后的人设正文」组装完整的 system prompt（重写只改表述，不改含义）。
    """
    return (rewritten_persona_text.strip() + _SYSTEM_PROMPT_TAIL).strip()


# 用于 LLM 重写人设的指令（不改变含义，仅换一种说法，减少模板感）
REWRITE_PERSONA_INSTRUCTION = """下面是一份「用户画像」的原始描述（用于后续让模型代入该人设填写问卷）。请用 2～4 段自然语言重写这份画像，要求：
1. 保持所有事实与含义完全不变（性别、年龄、职业、收入、常住地、四项心理动机等）。
2. 仅换一种说法、换一种口吻或句式，使读起来像另一份对同一人的描述。
3. 不要增加或删减任何信息，不要改变任何结论（如“节俭”“追求性价比”“渴望互动”等必须保留）。
4. 直接输出重写后的画像正文，不要输出 JSON，不要加“重写如下”等前缀。"""

# 重写时随机抽取的风格提示，使每次请求略有不同，降低缓存命中导致输出完全一致的概率
REWRITE_STYLE_HINTS = [
    "本次重写请偏口语化、像在向朋友介绍自己。",
    "本次重写请偏书面、像问卷说明里的受访者简介。",
    "本次重写请偏简短、用短句。",
    "本次重写请偏细腻、适当展开心理动机。",
    "本次重写请换一种段落顺序或句式结构。",
]


def rewrite_persona_with_llm(
    client: OpenAI,
    persona: dict,
    model: str = "deepseek-chat",
    max_retries: int = 3,
    base_delay: float = 3.0,
) -> Optional[str]:
    """
    用 LLM 将当前人设描述重写为同义不同表述，减少固定模板感，便于填答时同一类型下作答更分散。
    每次请求加入随机风格提示与请求标识，降低缓存命中导致重写完全一致的概率。
    """
    original_text = _get_persona_description_only(persona)
    style_hint = random.choice(REWRITE_STYLE_HINTS)
    # 请求标识使 user 内容每次不同，减少 CDN/API 缓存返回相同结果的可能
    request_id = f"{random.randint(10000, 99999)}"
    user_content = f"{REWRITE_PERSONA_INSTRUCTION}\n\n【补充】{style_hint}\n\n---\n请重写以下画像：\n\n{original_text}\n\n[req:{request_id}]"

    messages = [
        {"role": "system", "content": "你是一个将用户画像改写成同义不同表述的助手，严格保持事实与含义不变。"},
        {"role": "user", "content": user_content},
    ]

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.85,
                max_tokens=600,
            )
            out = (response.choices[0].message.content or "").strip()
            # 去掉可能被模型回显的请求标识
            if "[req:" in out:
                out = out.split("[req:")[0].strip()
            if out:
                return out
        except Exception as e:
            err = str(e).lower()
            if "rate_limit" in err or "429" in err:
                wait = base_delay * (2 ** (attempt - 1))
                logger.warning(f"  [重写人设 重试 {attempt}/{max_retries}] 频率限制，等待 {wait:.0f}s...")
                time.sleep(wait)
            else:
                logger.warning(f"  [重写人设 重试 {attempt}/{max_retries}] {e}")
    logger.warning("  人设重写失败，将使用原始模板。")
    return None


# ============================================================
# § 问卷题目文本（User Message）
# ============================================================

SURVEY_QUESTIONS = """
【问卷背景说明】本问卷假设的演出均在天津举办；以下题目中的「该城市」「演出举办城市」「目的地城市」均指天津。

请根据你的人设，回答以下所有问题，并以 JSON 格式输出全部答案。

=== 一、筛选题（所有人必填）===

Q1（您是否有线下观演的经历？）
  可选值（严格照抄其中一项）:
    "从未有过" | "1-3次" | "4-6次" | "7次及以上"

Q2（您是否有之后打算线下观演的计划？）
  可选值: "有" | "否" | "暂时不确定"

Q3（你是否有长期支持的偶像/乐队/歌手？）
  可选值: "无" | "1位" | "2-3位" | "4位及以上"

=== 二、偶像消费题（Q3 不为"无"时必填，否则填 null）===

Q4（除演出票外，还为偶像购买过哪些产品？多选，返回 JSON 数组；Q3="无" 时填 null）
  可选项列表:
    ["无", "实体专辑/数字音乐", "周边产品（海报/手办/应援物等）", "代言产品",
     "线上会员/直播打赏/应援打榜", "粉丝团/后援会相关费用", "其他"]

Q8（为偶像购买周边的月均金额？Q3="无" 时填 null）
  可选值: "50元及以下" | "50-200元" | "201-500元" | "501-2000元" | "2001元及以上" | null

=== 三、观演行为题（Q1 不为"从未有过"时必填，否则填 null）===

Q5（主要通过哪些渠道获取演出信息？多选，返回 JSON 数组；Q1="从未有过" 时填 null）
  可选项列表:
    ["微博/超话/粉丝群", "抖音/快手/小红书", "B站/视频号",
     "大麦/猫眼/票星球等票务平台", "朋友圈/朋友推荐",
     "偶像或乐队官方账号/工作室", "演出主办方/场馆官方宣传", "其他"]

Q6（一般优先选择哪个档次的座位？Q1="从未有过" 时填 null）
  可选值: "高端档（内场）" | "进阶档（优选看台）" | "基础档（普通看台）" | null

Q7（平均每场非门票类消费（交通/住宿/餐饮/周边等）？Q1="从未有过" 时填 null）
  可选值: "200元及以下" | "200-500元" | "501-1000元" | "1001-2000元" | "2001元及以上" | null

=== 四、态度量表题（天津场演出，所有人必填，整数 1-5，1=非常不同意，5=非常同意）===

Scale_1_1: 我在社交媒体（微博、抖音、小红书等）上频繁刷到关于本次演出的相关信息
Scale_1_2: 粉丝圈内高质量的安利内容（如视频剪辑、深度乐评）激发了我去现场的欲望
Scale_1_3: 粉丝社群/超话里热烈的抢票和出行讨论氛围，增强了我的参与意愿
Scale_2_1: 我觉得这位偶像/艺人就像我的好朋友或亲人一样
Scale_2_2: 我对这位偶像的作品、舞台风格和过往经历了如指掌
Scale_2_3: 我会主动关注这位偶像的社交媒体动态和最新消息
Scale_3_1: 去演出举办城市旅游，本身就在我的计划之中
Scale_3_2: 该城市独特的美食、景点或城市文化对我很有吸引力
Scale_3_3: 我认为该城市对游客是友好且服务周到的
Scale_4_1: 亲临现场看到偶像，能带给我线上观看无法替代的满足感和幸福感
Scale_4_2: 我渴望通过线下观演，获得那种具有强烈"临场感"和"真实感"的情感连接
Scale_4_3: 观看这场演出是我释放日常压力、获得积极情绪充电的重要方式
Scale_5_1: 在现场与成千上万人一起欢呼、合唱，能让我产生强烈的集体归属感
Scale_5_2: 作为粉丝群体的一员参与这场盛会，对我确认自己的"粉丝"身份很重要
Scale_5_3: 我渴望融入现场氛围，享受因为共同喜爱而产生的默契与共鸣
Scale_6_1: 我乐于参与从抢票、筹备应援物到奔赴现场的完整仪式过程
Scale_6_2: 按照粉丝社群的惯例参与现场互动（如跟唱、举灯牌、应援色），能让我获得强烈的意义感
Scale_6_3: 成功完成这次跨城观演之旅，对我来说是一次重要的自我实现和"成就打卡"
Scale_7_1: 考虑到交通、住宿和门票的总花费，我觉得为一场演出付出如此高的金钱成本不太值得
Scale_7_2: 协调出行时间、向公司/学校请假以及规划行程需要投入太多精力，这让我感到负担
Scale_7_3: 我担心演出期间，目的地城市的住宿、餐饮等价格溢价过高，导致严重超出预算
Scale_8_1: 只要有机会（如抢到票），我一定会去现场观看这次演出
Scale_8_2: 即使面临时间和金钱的压力，我也会优先考虑安排行程去参加这次演出
Scale_8_3: 我会向身边的朋友或粉丝社群强烈推荐去现场观看这场演出
Scale_9_1: 除了演出门票，我愿意在该城市停留更多天数进行旅游消费（住宿、餐饮）
Scale_9_2: 我愿意购买演出相关的周边产品或纪念品
Scale_9_3: 我愿意为了更好的观演体验（如更好的座位、更好的酒店）花更多的钱

=== 五、离散选择题 Conjoint（所有人必填）===
演唱会背景：某知名明星，天津奥林匹克中心体育场，周六 19:30-21:30，演出约120分钟。
票价含义：380元=普通看台（视野良好无遮挡）；680元=优选看台（近距离，音效更佳）；
         1280元=内场区域（沉浸式，最近距离）。
互动含义：无互动=仅观演；轻度互动=有概率上大屏被明星看到；深度互动=有概率被选中现场点歌互动。
文旅含义：无文旅服务=仅票务；单景点票=天津之眼或海河游船单张；双景点&酒店8折=两景点联票+指定酒店8折券。

Q10_1（第一组，从下列中选最愿意购买的一个，严格照抄选项文字）:
  "A: 380元（普通看台）+ 无互动 + 无文旅服务"
  "B: 680元（优选看台）+ 轻度互动（上大屏）+ 双景点&酒店8折"
  "C: 1280元（内场）+ 深度互动（点歌）+ 单景点票"
  "均不购买"

Q10_2（第二组，从下列中选最愿意购买的一个，严格照抄选项文字）:
  "A: 380元（普通看台）+ 轻度互动（上大屏）+ 单景点票"
  "B: 680元（优选看台）+ 深度互动（点歌）+ 无文旅服务"
  "C: 1280元（内场）+ 无互动 + 双景点&酒店8折"
  "均不购买"

Q10_3（第三组，从下列中选最愿意购买的一个，严格照抄选项文字）:
  "A: 380元（普通看台）+ 深度互动（点歌）+ 双景点&酒店8折"
  "B: 680元（优选看台）+ 无互动 + 单景点票"
  "C: 1280元（内场）+ 轻度互动（上大屏）+ 无文旅服务"
  "均不购买"

=== 六、不满因素题（所有人必填，最多选3项）===
Q_dissatisfaction（目前线下演出最让您感到不满的因素是？最多选3项，返回 JSON 数组）
  可选项列表（严格照抄，可选 1～3 项）:
    ["抢票机制不透明/黄牛横行", "现场秩序混乱/安保不力", "交通疏导不畅/散场难",
     "餐饮等配套设施物价过高", "演出内容缩水/音响效果差"]

输出一个合法 JSON，包含上述所有键：Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8,
Scale_1_1 ~ Scale_9_3（共27个），Q10_1, Q10_2, Q10_3，Q_dissatisfaction。
"""


# ============================================================
# § 模块三：调用 LLM API（带指数退避重试）
# ============================================================

def call_llm_api(
    client: OpenAI,
    system_prompt: str,
    model: str = "gpt-4o-mini",
    max_retries: int = 4,
    base_delay: float = 5.0,
) -> Optional[dict]:
    """
    调用 LLM API，强制以 json_object 格式返回问卷答案。

    Parameters
    ----------
    client      : 已初始化的 OpenAI 客户端
    system_prompt : 由 build_system_prompt() 生成的角色设定指令
    model       : 模型名称
    max_retries : 最大重试次数（遇到频率限制或网络错误时）
    base_delay  : 初始等待时长（秒），每次失败后指数翻倍

    Returns
    -------
    dict | None
        成功时返回解析后的 JSON 字典，失败超限后返回 None。
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": SURVEY_QUESTIONS},
    ]

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.85,
                max_tokens=1800,
            )
            raw = response.choices[0].message.content
            return json.loads(raw)

        except json.JSONDecodeError as e:
            logger.warning(f"  [重试 {attempt}/{max_retries}] JSON 解析失败: {e}")

        except Exception as e:
            err = str(e).lower()
            if "rate_limit" in err or "429" in err or "too many requests" in err:
                wait = base_delay * (2 ** (attempt - 1))
                logger.warning(f"  [重试 {attempt}/{max_retries}] 触发频率限制，等待 {wait:.0f}s...")
                time.sleep(wait)
            elif "timeout" in err or "connection" in err:
                wait = base_delay
                logger.warning(f"  [重试 {attempt}/{max_retries}] 网络超时，等待 {wait:.0f}s...")
                time.sleep(wait)
            else:
                logger.error(f"  [重试 {attempt}/{max_retries}] API 异常: {e}")
                if attempt == max_retries:
                    raise

    logger.error("  已达最大重试次数，本条记录跳过。")
    return None


# ============================================================
# § 工具函数：合并画像元数据与 LLM 答案为一行记录 + 增量保存
# ============================================================

def _save_progress(output_path: str, existing_records: list[dict], records: list[dict]) -> None:
    """将已有 + 当前记录写入 CSV（列顺序与最终一致），用于断点前增量保存。"""
    if not existing_records and not records:
        return
    all_records = existing_records + records
    df = pd.DataFrame(all_records)
    meta_cols = [c for c in df.columns if c.startswith("_meta_")]
    demo_cols = [c for c in df.columns if c.startswith("Q11_")]
    other_cols = [c for c in df.columns if c not in meta_cols and c not in demo_cols]
    df = df[other_cols + demo_cols + meta_cols]
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


def _flatten_record(persona: dict, llm_answers: dict) -> dict:
    """
    将 LLM 返回的 JSON 与画像元数据合并为一条 DataFrame 行。
    - 多选题（list 类型）以竖线 '|' 拼接为字符串
    - Q11.x 人口统计题直接由 persona 覆盖，保证数据一致性
    - 以 '_meta_' 前缀字段保留画像动机标签，便于后续验证
    """
    record: dict = {}

    for key, val in llm_answers.items():
        record[key] = "|".join(str(v) for v in val) if isinstance(val, list) else val

    # 人口统计题：由 persona 直接决定，避免 LLM 幻觉
    record["Q11_1_gender"]    = persona["gender"]
    record["Q11_2_age_range"] = persona["age_range"]
    record["Q11_3_occupation"] = persona["occupation"]
    record["Q11_4_income"]    = persona["income"]
    record["Q11_5_residence"] = persona["residence_detail"]

    # 画像动机元数据
    record["_meta_social_drive"]          = persona["social_drive"]
    record["_meta_consumption_pref"]      = persona["consumption_pref"]
    record["_meta_interaction_obsession"] = persona["interaction_obsession"]
    record["_meta_tourism_energy"]        = persona["tourism_energy"]
    record["_meta_residence_type"]        = persona["residence_type"]

    return record


# ============================================================
# § 模块四：主循环
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Z世代天津线下观演调研 — 合成问卷数据生成器",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n",        type=int,   default=50,
                        help="生成问卷份数")
    parser.add_argument("--model",    type=str,   default="deepseek-chat",
                        help="LLM 模型名称（DeepSeek: deepseek-chat / deepseek-reasoner）")
    parser.add_argument("--api-key",  type=str,   default=None,
                        help="DeepSeek API Key（也可设环境变量 DEEPSEEK_API_KEY）")
    parser.add_argument("--base-url", type=str,   default="https://api.deepseek.com",
                        help="API Base URL（默认 DeepSeek，也可替换为其他兼容端点）")
    parser.add_argument("--output",   type=str,   default="synthetic_survey_data.csv",
                        help="输出 CSV 文件路径")
    parser.add_argument("--resume",   action="store_true",
                        help="断点恢复：若 --output 文件已存在，则只生成不足部分使总行数达到 --n，再合并写入")
    parser.add_argument("--append-to", type=str,   default=None, metavar="PATH",
                        help="已有 CSV 路径：将本次生成的记录追加到该文件后再写入 --output；若未指定 --output 则默认写入同一文件")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="每次 API 调用之间的间隔秒数（防止频率限制）")
    parser.add_argument("--seed",     type=int,   default=None,
                        help="随机种子（设置后可复现画像采样结果）")
    parser.add_argument("--rewrite-persona", dest="rewrite_persona", action="store_true", default=True,
                        help="先用 LLM 重写人设再填问卷，减少模板感（默认开启）")
    parser.add_argument("--no-rewrite-persona", dest="rewrite_persona", action="store_false",
                        help="关闭人设重写，直接使用模板人设填问卷")
    parser.add_argument("--tianjin-only", dest="tianjin_only", action="store_true", default=False,
                        help="仅生成天津样本：常住地限为天津本地，地址从天津市内区中随机")
    parser.add_argument("--tianjin-ratio", type=float, default=None, metavar="P",
                        help="天津/非天津比例：每份样本以概率 P 为天津、1-P 为非天津（如 --n 500 --tianjin-ratio 0.35 得约 35%% 天津）。与 --tianjin-only 同时给出时以本参数为准")
    parser.add_argument("--cluster-sampling", dest="cluster_sampling", action="store_true", default=False,
                        help="整群抽样：天津从全市所有区中抽 6 区，非天津从全国所有省中抽 6 省，仅从抽中群内生成常住地")
    parser.add_argument("--tianjin-clusters", type=int, default=6,
                        help="整群抽样时天津抽取的区数（从全市 16 区中抽，默认 6）")
    parser.add_argument("--province-clusters", type=int, default=6,
                        help="整群抽样时非天津从全国省中抽取的省数（默认 6）；省内市任选。指定 --provinces 时本参数无效")
    parser.add_argument("--provinces", type=str, default=None, metavar="LIST",
                        help="整群抽样时指定外省列表，逗号分隔，如：云南省,江西省,福建省,海南省,宁夏回族自治区,西藏自治区。指定后仅从这些省中抽样（自动启用整群抽样）")
    args = parser.parse_args()

    # 续写模式：未显式指定 --output 时，默认写入与 --append-to 相同路径
    if args.append_to is not None and args.output == "synthetic_survey_data.csv":
        args.output = args.append_to

    if args.seed is not None:
        random.seed(args.seed)
        logger.info(f"随机种子已设置为 {args.seed}")

    # 指定 --provinces 时自动启用整群抽样
    specified_provinces: list[str] | None = None
    if args.provinces and args.provinces.strip():
        specified_provinces = [s.strip() for s in args.provinces.split(",") if s.strip()]
    use_cluster = args.cluster_sampling or bool(specified_provinces)

    global _RESIDENCE_POOL_RUN, _CLUSTER_RUN
    if use_cluster:
        _RESIDENCE_POOL_RUN, _CLUSTER_RUN = build_cluster_residence_pool(
            n_tianjin_clusters=args.tianjin_clusters,
            n_province_clusters=args.province_clusters,
            provinces=specified_provinces,
        )
        n_grps = len(_CLUSTER_RUN)  # 1 天津 + 若干外省群
        if specified_provinces:
            logger.info(
                f"整群抽样已启用 | 天津抽 {len(_RESIDENCE_POOL_RUN['天津本地'])} 个区 | "
                f"指定外省 {len(specified_provinces)} 个：{', '.join(specified_provinces)} → 共 {n_grps} 群"
            )
        else:
            logger.info(
                f"整群抽样已启用 | 天津抽 {len(_RESIDENCE_POOL_RUN['天津本地'])} 个区 | "
                f"全国省中抽 {args.province_clusters} 个省 → 共 {n_grps} 群，非天津 only 时按群等权抽样"
            )
    else:
        _RESIDENCE_POOL_RUN = None
        _CLUSTER_RUN = None

    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "未找到 API Key！请通过 --api-key 参数或设置环境变量 DEEPSEEK_API_KEY 提供。"
        )

    # 初始化客户端（OpenAI SDK 兼容 DeepSeek API）
    client = OpenAI(api_key=api_key, base_url=args.base_url)

    # 断点恢复：若 --resume 且输出文件已存在，只生成不足部分
    existing_records: list[dict] = []
    n_to_generate = args.n
    if args.resume:
        out_path = os.path.abspath(args.output)
        if os.path.isfile(out_path):
            df_existing = pd.read_csv(out_path, encoding="utf-8-sig")
            existing_count = len(df_existing)
            need = args.n - existing_count
            if need <= 0:
                logger.info(f"断点恢复：{args.output} 已有 {existing_count} 行，已达目标 --n {args.n}，无需生成。")
                return
            existing_records = df_existing.to_dict("records")
            n_to_generate = need
            logger.info(f"断点恢复：已有 {existing_count} 行，再生成 {n_to_generate} 份至目标 {args.n}。")
        else:
            logger.info("断点恢复：输出文件不存在，将从头生成。")

    records: list[dict] = []
    success_count = 0
    fail_count    = 0

    p_tianjin = (max(0.0, min(1.0, args.tianjin_ratio)) if args.tianjin_ratio is not None else None)
    if args.append_to is not None:
        logger.info(f"续写模式：将追加到 {args.append_to}，结果写入 {args.output}")
    if p_tianjin is not None:
        logger.info(
            f"开始生成 {args.n} 份合成问卷 | 模型: {args.model} | 输出: {args.output} | "
            f"人设重写: {'开' if args.rewrite_persona else '关'} | 天津比例: {p_tianjin:.0%} | "
            f"整群抽样: {'是' if args.cluster_sampling else '否'}"
        )
    else:
        logger.info(
            f"开始生成 {args.n} 份合成问卷 | 模型: {args.model} | 输出: {args.output} | "
            f"人设重写: {'开' if args.rewrite_persona else '关'} | 天津限定: {'是' if args.tianjin_only else '否'} | "
            f"整群抽样: {'是' if args.cluster_sampling else '否'}"
        )
    logger.info("-" * 60)

    for i in range(1, n_to_generate + 1):
        logger.info(f"[{i:>4}/{n_to_generate}] 正在处理...")
        tianjin_this = (random.random() < p_tianjin) if p_tianjin is not None else args.tianjin_only
        persona = generate_persona(tianjin_only=tianjin_this)
        if args.rewrite_persona:
            rewritten = rewrite_persona_with_llm(client, persona, model=args.model)
            system_prompt = build_system_prompt_from_rewritten(rewritten) if rewritten else build_system_prompt(persona)
        else:
            system_prompt = build_system_prompt(persona)
        llm_answers = call_llm_api(client, system_prompt, model=args.model)

        if llm_answers is None:
            fail_count += 1
            logger.warning(f"  → 第 {i} 份失败，跳过。")
            continue

        record = _flatten_record(persona, llm_answers)
        records.append(record)
        success_count += 1
        # 每成功一条即写入，崩溃或中断后可用 --resume 继续
        _save_progress(args.output, existing_records, records)

        logger.info(
            f"  → 成功 ✓ | 画像: {persona['gender']} / {persona['age_occ_group']} / "
            f"{persona['income']} / {persona['consumption_pref']}"
        )

        if i < n_to_generate:
            time.sleep(args.interval)

    logger.info("-" * 60)

    if not records and not existing_records:
        logger.error("没有生成任何有效数据，请检查 API Key 和网络连接。")
        return

    # 断点恢复时合并已有 + 本次新生成
    if existing_records:
        all_records = existing_records + records
        df = pd.DataFrame(all_records)
        logger.info(f"已合并：原有 {len(existing_records)} 行 + 本次 {len(records)} 行 = {len(df)} 行")
    else:
        df = pd.DataFrame(records)

    # 规范化列顺序：人口统计列置后，元数据列置最后
    meta_cols = [c for c in df.columns if c.startswith("_meta_")]
    demo_cols = [c for c in df.columns if c.startswith("Q11_")]
    other_cols = [c for c in df.columns if c not in meta_cols and c not in demo_cols]
    df = df[other_cols + demo_cols + meta_cols]

    # 续写：读取已有 CSV，校验列一致后拼接，再写入
    if args.append_to is not None:
        path_append = os.path.abspath(args.append_to)
        if os.path.isfile(path_append):
            df_existing = pd.read_csv(path_append, encoding="utf-8-sig")
            if set(df_existing.columns) != set(df.columns):
                logger.error(
                    "已有文件与本次生成的列不一致，无法安全追加。请勿使用 --append-to 或保证文件由本脚本生成。"
                )
                return
            df = df.reindex(columns=df_existing.columns)
            df = pd.concat([df_existing, df], ignore_index=True)
            logger.info(f"已续写：本次新增 {success_count} 份，合计 {len(df)} 行")
        else:
            logger.warning(f"--append-to 指向的文件不存在: {path_append}，将仅保存本次生成的 {len(df)} 行")

    df.to_csv(args.output, index=False, encoding="utf-8-sig")

    logger.info(f"完成！本次成功: {success_count} 份 | 失败: {fail_count} 份 | "
                f"成功率: {success_count / n_to_generate * 100:.1f}%")
    logger.info(f"数据已保存至: {args.output}  （{len(df.columns)} 列 × {len(df)} 行）")


if __name__ == "__main__":
    main()
