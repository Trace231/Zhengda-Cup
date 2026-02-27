"""
Z世代天津线下观演调研 — 合成问卷数据生成脚本
================================================
依赖库：
    pip install openai pandas python-docx

运行示例：
    # 最简运行（需已设置环境变量 DEEPSEEK_API_KEY）
    python data_pipeline.py --n 100

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
    ("17-21岁学生",       0.35),
    ("22-26岁职场新人",   0.45),
    ("27-31岁资深职场人", 0.20),
]

PRIOR_INCOME = [
    ("1000元及以下",  0.10),
    ("1001-3000元",  0.35),
    ("3001-6000元",  0.25),
    ("6001-10000元", 0.20),
    ("10000元以上",  0.10),
]

PRIOR_RESIDENCE = [
    ("天津本地",     0.30),
    ("京津冀近途",   0.45),
    ("其他省份远途", 0.25),
]


# ============================================================
# § 内在条件概率 P(I|E)
# ============================================================

# 社交驱动 × 年龄职业
SOCIAL_DRIVE_BY_AGE: dict[str, list] = {
    "17-21岁学生":       [("饭圈打卡", 0.50), ("精致出片", 0.35), ("解压放松", 0.15)],
    "22-26岁职场新人":   [("饭圈打卡", 0.30), ("精致出片", 0.40), ("解压放松", 0.30)],
    "27-31岁资深职场人": [("饭圈打卡", 0.10), ("精致出片", 0.20), ("解压放松", 0.70)],
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
    文旅精力联合条件概率：P(精力 | 常住地, 年龄职业)
    远途 + 资深职场人 → 精力匮乏概率最高（85%），
    本地/近途 + 学生    → 精力充沛概率最高（85%）。
    """
    is_remote = (residence == "其他省份远途")
    if "学生" in age_occ:
        if is_remote:
            return [("精力充沛", 0.55), ("精力一般", 0.30), ("精力匮乏", 0.15)]
        else:
            return [("精力充沛", 0.85), ("精力一般", 0.10), ("精力匮乏", 0.05)]
    elif "职场新人" in age_occ:
        if is_remote:
            return [("精力充沛", 0.10), ("精力一般", 0.20), ("精力匮乏", 0.70)]
        else:
            return [("精力充沛", 0.50), ("精力一般", 0.35), ("精力匮乏", 0.15)]
    else:  # 27-31岁资深职场人
        if is_remote:
            return [("精力充沛", 0.05), ("精力一般", 0.10), ("精力匮乏", 0.85)]
        else:
            return [("精力充沛", 0.30), ("精力一般", 0.40), ("精力匮乏", 0.30)]


def _weighted_choice(pairs: list) -> str:
    """按权重随机从 [(值, 概率), ...] 中抽取一个值"""
    values, weights = zip(*pairs)
    return random.choices(values, weights=weights, k=1)[0]


# ============================================================
# § 模块一：生成用户画像字典
# ============================================================

# 常住地到具体城市/区的映射候选池
_RESIDENCE_POOL: dict[str, list] = {
    "天津本地": [
        "天津市南开区", "天津市河西区", "天津市河东区",
        "天津市和平区", "天津市滨海新区", "天津市西青区",
    ],
    "京津冀近途": [
        "北京市朝阳区", "北京市海淀区", "北京市丰台区",
        "河北省廊坊市", "河北省保定市", "河北省石家庄市",
    ],
    "其他省份远途": [
        "上海市浦东新区", "广东省广州市", "浙江省杭州市",
        "四川省成都市", "山东省济南市", "湖北省武汉市", "江苏省南京市",
    ],
}

# 年龄职业段到具体职业的细化映射
_OCCUPATION_POOL: dict[str, tuple[list, list]] = {
    "17-21岁学生": (
        ["高中生", "本科生/专科生", "研究生"],
        [0.20, 0.65, 0.15],
    ),
    "22-26岁职场新人": (
        ["企业/单位在职人员（工作1-3年）", "自由职业者", "待业/备考"],
        [0.70, 0.20, 0.10],
    ),
    "27-31岁资深职场人": (
        ["企业/单位在职人员（工作4-8年）", "自由职业者"],
        [0.80, 0.20],
    ),
}

_AGE_RANGE_LABEL: dict[str, str] = {
    "17-21岁学生":       "17-21岁（2005-2009年出生）",
    "22-26岁职场新人":   "22-26岁（2000-2004年出生）",
    "27-31岁资深职场人": "27-31岁（1995-1999年出生）",
}


def generate_persona() -> dict:
    """
    基于贝叶斯先验 + 条件概率采样，生成一份逻辑自洽的用户画像字典。

    Returns
    -------
    dict
        包含外在人口属性（gender / age_occ_group / occupation / income / residence_*）
        与内在心理动机（social_drive / consumption_pref / interaction_obsession / tourism_energy）。
    """
    # --- 外在属性：独立先验采样 ---
    gender    = _weighted_choice(PRIOR_GENDER)
    age_occ   = _weighted_choice(PRIOR_AGE_OCC)
    income    = _weighted_choice(PRIOR_INCOME)
    residence = _weighted_choice(PRIOR_RESIDENCE)

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
        "residence_detail":     random.choice(_RESIDENCE_POOL[residence]),
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


def build_system_prompt(persona: dict) -> str:
    """
    将画像字典转化为极具代入感的角色扮演指令，
    让 LLM 以该人设的逻辑和情感填写问卷。
    """
    prompt = f"""你是一名{persona['gender']}生，年龄段为{persona['age_range']}，
职业是{persona['occupation']}，月可支配收入为{persona['income']}，
常住地为{persona['residence_detail']}（属于"{persona['residence_type']}"群体）。

【你的核心性格与动机】
- 社交驱动类型：{_SOCIAL_DRIVE_DESC[persona['social_drive']]}
- 消费偏好类型：{_CONSUMPTION_DESC[persona['consumption_pref']]}
- 互动执念程度：{_INTERACTION_DESC[persona['interaction_obsession']]}
- 文旅精力状态：{_ENERGY_DESC[persona['tourism_energy']]}

【角色扮演要求】
请完全代入上述人设，以第一人称视角，基于你的收入水平、生活状态和心理动机，
真实填写以下"Z世代天津线下观演调研"问卷。你的回答必须与人设逻辑高度自洽，
不可与收入层级、职业现实和核心动机产生明显矛盾。
特别地，「情感体验动机」相关题目（亲临现场的满足感、临场感与真实感、释放压力与情绪充电）
测量的是同一维度，请保持态度一致，均围绕“现场观演带来的情感价值”在 1–5 之间给出连贯评分。

【输出格式要求】
严格输出一个合法的 JSON 对象，键名与题号完全一致，不得在 JSON 之外输出任何文字或解释。"""
    return prompt.strip()


# ============================================================
# § 问卷题目文本（User Message）
# ============================================================

SURVEY_QUESTIONS = """
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

=== 四、态度量表题（所有人必填，整数 1-5，1=非常不同意，5=非常同意）===

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

输出一个合法 JSON，包含上述所有键：Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8,
Scale_1_1 ~ Scale_9_3（共27个），Q10_1, Q10_2, Q10_3。
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
# § 工具函数：合并画像元数据与 LLM 答案为一行记录
# ============================================================

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
    parser.add_argument("--interval", type=float, default=1.0,
                        help="每次 API 调用之间的间隔秒数（防止频率限制）")
    parser.add_argument("--seed",     type=int,   default=None,
                        help="随机种子（设置后可复现画像采样结果）")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        logger.info(f"随机种子已设置为 {args.seed}")

    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "未找到 API Key！请通过 --api-key 参数或设置环境变量 DEEPSEEK_API_KEY 提供。"
        )

    # 初始化客户端（OpenAI SDK 兼容 DeepSeek API）
    client = OpenAI(api_key=api_key, base_url=args.base_url)

    records: list[dict] = []
    success_count = 0
    fail_count    = 0

    logger.info(f"开始生成 {args.n} 份合成问卷 | 模型: {args.model} | 输出: {args.output}")
    logger.info("-" * 60)

    for i in range(1, args.n + 1):
        logger.info(f"[{i:>4}/{args.n}] 正在处理...")

        persona       = generate_persona()
        system_prompt = build_system_prompt(persona)
        llm_answers   = call_llm_api(client, system_prompt, model=args.model)

        if llm_answers is None:
            fail_count += 1
            logger.warning(f"  → 第 {i} 份失败，跳过。")
            continue

        record = _flatten_record(persona, llm_answers)
        records.append(record)
        success_count += 1

        logger.info(
            f"  → 成功 ✓ | 画像: {persona['gender']} / {persona['age_occ_group']} / "
            f"{persona['income']} / {persona['consumption_pref']}"
        )

        if i < args.n:
            time.sleep(args.interval)

    logger.info("-" * 60)

    if not records:
        logger.error("没有生成任何有效数据，请检查 API Key 和网络连接。")
        return

    df = pd.DataFrame(records)

    # 规范化列顺序：人口统计列置后，元数据列置最后
    meta_cols = [c for c in df.columns if c.startswith("_meta_")]
    demo_cols = [c for c in df.columns if c.startswith("Q11_")]
    other_cols = [c for c in df.columns if c not in meta_cols and c not in demo_cols]
    df = df[other_cols + demo_cols + meta_cols]

    df.to_csv(args.output, index=False, encoding="utf-8-sig")

    logger.info(f"完成！成功: {success_count} 份 | 失败: {fail_count} 份 | "
                f"成功率: {success_count / args.n * 100:.1f}%")
    logger.info(f"数据已保存至: {args.output}  （{len(df.columns)} 列 × {len(df)} 行）")


if __name__ == "__main__":
    main()
