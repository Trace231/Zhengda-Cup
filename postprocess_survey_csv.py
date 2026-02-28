#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
后处理 data_pipeline 生成的问卷 CSV：删除 AI 生成时的种子痕迹。
- 删除所有 _meta_* 列（画像动机标签：饭圈打卡、极致前排控、强互动执念 等）
- 可选：在其余文本列中抹掉上述种子短语（防止开放式题里被 LLM 复述）

用法：
    python postprocess_survey_csv.py synthetic_survey_data.csv -o survey_clean.csv
    python postprocess_survey_csv.py survey_300.csv -o survey_300_clean.csv --strip-phrases
"""

import argparse
import re
import pandas as pd
from pathlib import Path

# 生成时使用的种子标签（用于 --strip-phrases 时从文本中抹掉）
SEED_PHRASES = [
    "饭圈打卡", "精致出片", "解压放松",
    "抠门陪伴党", "理性消费者", "极致前排控",
    "强互动执念", "弱互动执念",
    "精力充沛", "精力一般", "精力匮乏",
    "天津本地", "京津冀近途", "中程较近", "其他省份远途",
]


def drop_meta_columns(df: pd.DataFrame) -> pd.DataFrame:
    """删除所有以 _meta_ 开头的列。"""
    meta_cols = [c for c in df.columns if c.startswith("_meta_")]
    if not meta_cols:
        return df
    return df.drop(columns=meta_cols)


def strip_seed_phrases_from_cell(val: str) -> str:
    """在单个字符串中抹掉种子短语，并整理竖线分隔的空位。"""
    if pd.isna(val) or not isinstance(val, str):
        return val
    s = val
    for phrase in SEED_PHRASES:
        s = s.replace(phrase, "")
    # 合并多余竖线并去掉首尾竖线
    s = re.sub(r"\|+", "|", s).strip("| ")
    return s if s else ""


def strip_phrases_in_df(df: pd.DataFrame) -> pd.DataFrame:
    """对所有 object/string 列做 strip_seed_phrases_from_cell。"""
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].astype(str).map(strip_seed_phrases_from_cell)
            # 把 "nan" 字符串还原为真正的空
            out.loc[out[col] == "nan", col] = ""
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="删除问卷 CSV 中的 AI 种子痕迹（_meta_ 列及可选文本中的种子短语）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", type=str, help="输入的 CSV 路径（data_pipeline 生成）")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="输出 CSV 路径；不填则覆盖输入文件（会先写临时再覆盖）")
    parser.add_argument("--strip-phrases", action="store_true",
                        help="在除 _meta_ 外的文本列中抹掉种子短语（如 极致前排控、饭圈打卡 等）")
    parser.add_argument("--no-drop-meta", action="store_true",
                        help="不删除 _meta_ 列（仅与 --strip-phrases 一起用时有意义）")
    args = parser.parse_args()

    path_in = Path(args.input)
    if not path_in.exists():
        raise FileNotFoundError(f"输入文件不存在: {path_in}")

    df = pd.read_csv(path_in, encoding="utf-8-sig")
    meta_cols = [c for c in df.columns if c.startswith("_meta_")]

    if not args.no_drop_meta:
        df = drop_meta_columns(df)
        print(f"已删除 {len(meta_cols)} 列 _meta_*: {meta_cols}")
    if args.strip_phrases:
        df = strip_phrases_in_df(df)
        print("已在文本列中抹掉种子短语。")

    out_path = Path(args.output) if args.output else path_in
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"已写入: {out_path}（共 {len(df)} 行, {len(df.columns)} 列）")


if __name__ == "__main__":
    main()
