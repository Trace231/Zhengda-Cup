# 数据生成「续写」功能 — 改动说明

## 功能说明

在已有 CSV 基础上**追加**本次生成的记录，而不是覆盖。例如：已有 300 条，再生成 200 条，得到 500 条并写回同一文件。

---

## 代码改动摘要

### 1. 新增命令行参数（`data_pipeline.py`）

- **`--append-to PATH`**：指定已有 CSV 路径。脚本会先读取该文件，将本次生成的记录拼接到其末尾，再写入 `--output`。
- 若只传 `--append-to` 而未传 `--output`，则**默认写入与 `--append-to` 相同路径**（即“原地续写”）。

### 2. 默认输出路径逻辑（`main()` 内，`parse_args` 之后）

```text
若 args.append_to 不为 None 且 args.output 为默认值 "synthetic_survey_data.csv"
  → 令 args.output = args.append_to
```

这样在只加 `--append-to survey_300.csv` 时，结果会写回 `survey_300.csv`。

### 3. 续写流程（在 `pd.DataFrame(records)` 且列顺序规范化之后）

1. **若未使用 `--append-to`**：与原先一致，仅将本次 `df` 写入 `args.output`。
2. **若使用了 `--append-to`**：
   - 若 `args.append_to` 指向的**文件存在**：
     - 用 `pd.read_csv(..., encoding="utf-8-sig")` 读入已有数据；
     - 校验：`set(已有列) == set(本次 df 列)`，不一致则报错并退出；
     - 按已有文件的列顺序重排本次 `df`，再 `pd.concat([已有, 本次], ignore_index=True)`；
     - 将合并后的 DataFrame 写入 `args.output`，并打日志「本次新增 X 份，合计 Y 行」。
   - 若**文件不存在**：打 warning，不读已有文件，仅将本次生成的 `df` 写入 `args.output`（相当于普通覆盖到该路径）。

### 4. 日志

- 开始时若传入 `--append-to`，会打印：`续写模式：将追加到 <path>，结果写入 <output>`。
- 成功续写后打印：`已续写：本次新增 X 份，合计 Y 行`。

---

## 使用示例

```bash
# 在 survey_300.csv 上再生成 200 条，结果写回 survey_300.csv（共 500 行）
python data_pipeline.py --n 200 --append-to survey_300.csv

# 续写到 survey_300.csv，但结果保存到新文件 survey_500.csv
python data_pipeline.py --n 200 --append-to survey_300.csv --output survey_500.csv
```

---

## 注意事项

- **列一致性**：`--append-to` 指向的 CSV 必须与本脚本当前版本生成的列完全一致（列名与列集相同），否则会报错并退出，避免产生错列对齐数据。
- **编码**：读写均使用 `utf-8-sig`，与现有行为一致。
- **文件不存在**：若 `--append-to` 指向的文件不存在，不会报错退出，只会 warning 并只保存本次生成的记录到 `args.output`。

---

## 涉及文件

- **仅修改**：`data_pipeline.py`（参数、默认输出逻辑、续写分支与写入）。
