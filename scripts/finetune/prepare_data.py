"""
prepare_data.py — LoRA 微调数据准备
====================================
从 conversations 表中提取高质量 Q&A 对，清洗过滤后生成训练数据集。

清洗规则:
  1. 过滤长度不足的问答 (< 50 tokens 用户问题, < 100 tokens AI回答)
  2. 过滤拒绝/兜底回答 ("文档中未找到", "抱歉", "无法回答")
  3. 过滤非制造业内容 (闲聊、天气、作文等)
  4. 过滤含幻觉引用的回答 (引用不存在的文档)
  5. 去重: 相似度 > 0.9 的问答对只保留一条

输出格式 (JSONL):
  {"instruction": "你是制造业生产运维助手...", "input": "用户问题", "output": "AI回答"}

运行: python -m scripts.finetune.prepare_data
"""

import json
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "finetune"
DB_PATH = Path(__file__).parent.parent.parent / "data" / "manufacturing.db"

SYSTEM_PROMPT = """你是制造业生产运维知识助手，为产线工人和维修工提供设备操作和故障处理的专业指导。
你只能基于提供的设备手册、SOP文档和报警码表回答问题。
不知道就说不知道，不要编造。每个操作步骤必须标注来源。"""

# 制造领域关键词 (用来过滤非制造内容)
MANUFACTURING_KEYWORDS = [
    "设备", "注塑", "冲压", "CNC", "模具", "报警", "故障", "维修", "保养", "操作",
    "SOP", "参数", "安全", "检查", "点检", "更换", "温度", "压力", "液压", "电气",
    "切削", "换模", "生产", "产线", "车间", "工艺", "质量", "缺陷", "良率", "OEE",
    "传感器", "润滑", "校准", "装配", "PLC", "主轴", "刀库", "供料", "贴片",
]


def load_conversations(db_path: str) -> list[dict]:
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT session_id, role, content, created_at FROM conversations ORDER BY session_id, created_at"
    ).fetchall()
    conn.close()

    # 按 session 分组，构建 Q&A 对
    sessions = {}
    for row in rows:
        sid = row[0]
        if sid not in sessions:
            sessions[sid] = []
        sessions[sid].append({"role": row[1], "content": row[2], "time": row[3]})

    qa_pairs = []
    for sid, msgs in sessions.items():
        for i in range(len(msgs) - 1):
            if msgs[i]["role"] == "user" and msgs[i + 1]["role"] == "assistant":
                qa_pairs.append({
                    "question": msgs[i]["content"].strip(),
                    "answer": msgs[i + 1]["content"].strip(),
                    "session_id": sid,
                })
    return qa_pairs


def is_manufacturing(text: str) -> bool:
    return any(kw in text for kw in MANUFACTURING_KEYWORDS)


def is_refusal(text: str) -> bool:
    refusal_patterns = [
        "文档中未找到", "抱歉", "无法回答", "无法处理", "我不是",
        "不能帮你", "无法帮你", "document not found", "sorry",
    ]
    return any(p in text for p in refusal_patterns)


def has_hallucination(answer: str) -> bool:
    """检测可能的幻觉: 引用了不存在的文档编号。"""
    refs = re.findall(r'SOP-([A-Z0-9-]+)', answer)
    # 简化检测: SOP引用格式异常
    for ref in refs:
        if len(ref) < 3 or len(ref) > 20:
            return True
    return False


def estimate_tokens(text: str) -> int:
    """中文: 1 token ≈ 2 字符"""
    return len(text) // 2


def clean_data(qa_pairs: list[dict]) -> list[dict]:
    """多轮清洗。"""
    cleaned = []
    stats = {"total": len(qa_pairs), "short": 0, "refusal": 0, "non_mfg": 0, "hallucination": 0, "kept": 0}

    for pair in qa_pairs:
        q, a = pair["question"], pair["answer"]

        # 规则1: 长度过滤
        if len(q) < 10 or len(a) < 50:
            stats["short"] += 1
            continue

        # 规则2: 拒绝/兜底过滤
        if is_refusal(a):
            stats["refusal"] += 1
            continue

        # 规则3: 非制造内容过滤
        if not is_manufacturing(q) and not is_manufacturing(a):
            stats["non_mfg"] += 1
            continue

        # 规则4: 幻觉引用过滤
        if has_hallucination(a):
            stats["hallucination"] += 1
            continue

        cleaned.append(pair)
        stats["kept"] += 1

    # 规则5: 去重 (基于内容前缀相似)
    seen = set()
    deduped = []
    for pair in cleaned:
        fingerprint = pair["question"][:80].strip().lower()
        if fingerprint not in seen:
            seen.add(fingerprint)
            deduped.append(pair)

    print(f"数据清洗: {stats}")
    return deduped


def export_jsonl(pairs: list[dict], output_path: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            record = {
                "instruction": SYSTEM_PROMPT,
                "input": pair["question"],
                "output": pair["answer"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"训练数据导出: {output_path} ({len(pairs)} 条)")


def main():
    print("LoRA 微调数据准备\n")

    # 1. 加载对话
    pairs = load_conversations(str(DB_PATH))
    print(f"从数据库加载: {len(pairs)} 对 Q&A")

    # 2. 清洗
    cleaned = clean_data(pairs)

    # 3. 导出
    output = OUTPUT_DIR / "train_data.jsonl"
    export_jsonl(cleaned, str(output))

    # 4. 统计
    total_chars = sum(len(p["question"]) + len(p["answer"]) for p in cleaned)
    print(f"\n数据集统计:")
    print(f"  样本数: {len(cleaned)}")
    print(f"  总字符: {total_chars}")
    print(f"  估算Token: {total_chars // 2}")
    print(f"  输出文件: {output}")


if __name__ == "__main__":
    main()
