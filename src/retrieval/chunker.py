"""
chunker.py — 语义分块 + 工业文档适配
====================================
角色：将原始文档按语义边界切分为可检索的 chunk。
      制造业文档有特殊结构 (参数表、SOP步骤、报警码表)，不能一刀切。

分块策略 (按文档内容类型自动选择):

1. 表格检测 → 表级分块
   触发条件: 检测到 |---| 或 tab 分隔的结构化数据
   处理: 整表作为一个 chunk，不切割
   示例:
   | 参数 | 值 | 单位 |
   | 料筒温度 | 200-280 | ℃ |
   | 注射压力 | 80-120 | MPa |
   → 整个表格一个 chunk，保证参数不被切散

2. SOP 步骤 → 步骤级分块
   触发条件: 检测到 "步骤1:" "Step 1:" "1." 等编号模式
   处理: 按编号边界切分，确保每个步骤完整
   示例:
   "步骤1: 关闭料筒加热开关，等待温度降至100℃以下
    步骤2: 松开模具固定螺栓..."
   → Chunk1: "步骤1: 关闭料筒加热..."
   → Chunk2: "步骤2: 松开模具..."

3. 报警码表 → 表级分块
   触发条件: 检测到 "报警码" "故障码" "Alarm Code" 等 header
   处理: 整表一个 chunk，每行一个报警码对应关系不拆散

4. 常规文本 → 语义分块
   触发条件: 不匹配上述任何特殊类型
   处理: 按段落/标题的语义边界切分，保留文档层级 (H1-H2-H3)
   chunk_size: 512 tokens (中文约 1000 字)
   overlap: 64 tokens (保证边界处的关键信息不丢失)

使用方式:
    from src.retrieval.chunker import DocumentChunker
    chunks = DocumentChunker().chunk(text="完整文档内容", title="注塑机操作手册")

注意事项:
- overlap 会导致同一段文字出现在两个 chunk 中，检索时会去重
- 表级分块的 chunk 可能很大 (几百个参数)，但完整性 > 大小
- 未来扩展: CAD 图纸 OCR 文本 → 绑定设备型号 → 图文关联检索
"""

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Chunk:
    """分块结果。"""
    chunk_id: str          # 唯一ID，如 "doc001_chunk_003"
    content: str           # 分块文本
    chunk_index: int       # 序号，从0开始
    chunk_type: str        # "table" / "sop_step" / "alarm_table" / "semantic"
    heading: str = ""      # 所属章节标题 (如 "第三章 设备参数")
    doc_title: str = ""    # 文档标题
    metadata: dict = field(default_factory=dict)  # 额外元数据 (如设备型号)


class DocumentChunker:
    """文档智能分块器。"""

    # 表格检测模式
    TABLE_PATTERN = re.compile(r'\|.*\|.*\n\s*\|[-|]+\|', re.MULTILINE)
    # SOP 步骤检测
    STEP_PATTERN = re.compile(
        r'(?:步骤\s*\d+|Step\s*\d+|第[一二三四五六七八九十\d]+步)[\s:：.]',
        re.IGNORECASE
    )
    # 报警码检测
    ALARM_PATTERN = re.compile(
        r'(报警码|故障码|Alarm\s*Code|Error\s*Code|报警\s*代码|故障\s*代码)',
        re.IGNORECASE
    )

    # Markdown 标题
    MD_HEADER = re.compile(r'^#{1,4}\s+', re.MULTILINE)

    def __init__(
        self,
        chunk_size: int = 256,    # tokens，中文约 500 字
        chunk_overlap: int = 32,  # tokens，中文约 60 字
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _detect_content_type(self, text: str) -> str:
        """自动检测文档内容类型。

        返回: "table" | "sop_step" | "alarm_table" | "semantic"
        """
        if self.ALARM_PATTERN.search(text[:500]):
            return "alarm_table"
        if self.TABLE_PATTERN.search(text):
            return "table"
        if self.STEP_PATTERN.search(text):
            return "sop_step"
        return "semantic"

    def _split_by_md_headers(self, text: str) -> list[tuple[str, str]]:
        """按 Markdown 标题 (# ## ### ####) 切分，返回 [(标题, 正文), ...]。
        没有标题时返回 [("", text)]。
        """
        parts = re.split(r'^(#{1,4}\s+.+)$', text, flags=re.MULTILINE)
        sections = []
        current_heading = ""
        current_body = ""
        for part in parts:
            if re.match(r'^#{1,4}\s+', part):
                if current_body.strip():
                    sections.append((current_heading, current_body.strip()))
                current_heading = part
                current_body = ""
            else:
                current_body += part
        if current_body.strip() or current_heading:
            sections.append((current_heading, current_body.strip()))
        return sections or [("", text)]

    def _chunk_semantic(self, text: str) -> list[str]:
        """常规文本语义分块: 先按标题切分，再按段落 + 长度切分。"""
        char_limit = self.chunk_size * 2  # ~500 字 × 2 = ~1000 字符上限
        overlap = self.chunk_overlap * 2  # ~60 字重叠

        # 先按 Markdown 标题切大段
        sections = self._split_by_md_headers(text)
        all_chunks = []

        for heading, body in sections:
            # 按段落分割
            paragraphs = body.split('\n\n')
            current = heading + '\n' if heading else ""

            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue

                # 短行（可能是子标题），作为新 chunk 起点
                if len(para) < 60 and not para.endswith('。'):
                    if len(current) > len(heading) + 10:
                        all_chunks.append(current)
                    current = (heading + '\n' if heading else "") + para
                    continue

                combined = current + '\n\n' + para if (
                    current and current != (heading + '\n' if heading else "")
                ) else para
                if heading and current == heading + '\n':
                    combined = heading + '\n' + para

                if len(combined) <= char_limit:
                    current = combined
                else:
                    if current and len(current) > len(heading) + 10:
                        all_chunks.append(current)
                    # 如果单个段落超过上限，按字符滑动窗口切分
                    if len(para) > char_limit:
                        sub_chunks = self._split_long_para(para, heading, char_limit, overlap)
                        all_chunks.extend(sub_chunks)
                        current = heading + '\n' if heading else ""
                    else:
                        current = (heading + '\n' if heading else "") + para

            if current and len(current) > len(heading) + 5:
                all_chunks.append(current)

        return all_chunks or [text]

    def _split_long_para(self, text: str, heading: str, limit: int, overlap: int) -> list[str]:
        """超长段落按滑动窗口切分。"""
        chunks = []
        start = 0
        prefix = heading + '\n' if heading else ""
        while start < len(text):
            end = min(start + limit, len(text))
            chunk_text = prefix + text[start:end]
            chunks.append(chunk_text)
            if end >= len(text):
                break
            start = end - overlap
        return chunks

    def _chunk_by_steps(self, text: str) -> list[str]:
        """SOP 步骤级分块: 按步骤编号切分。

        示例:
            text = "步骤1: 关闭开关\n步骤2: 松开螺栓\n步骤3: 更换零件"
            → ["步骤1: 关闭开关", "步骤2: 松开螺栓", "步骤3: 更换零件"]
        """
        # 按步骤编号切分
        parts = re.split(
            r'\n(?=(?:步骤\s*\d+|Step\s*\d+|第[一二三四五六七八九十\d]+步))',
            text, flags=re.IGNORECASE
        )
        return [p.strip() for p in parts if p.strip()]

    def _chunk_table(self, text: str) -> list[str]:
        """表格级分块: 整表一个 chunk。"""
        return [text]  # 整表不分块

    def chunk(
        self,
        text: str,
        title: str = "",
        heading: str = "",
        equipment_model: str = "",
    ) -> list[Chunk]:
        """主分块入口。

        参数:
            text: 文档全文
            title: 文档标题
            heading: 当前章节标题
            equipment_model: 关联设备型号 (如 "海天MA1200")

        返回:
            分块列表

        示例:
            chunker = DocumentChunker()
            chunks = chunker.chunk(
                text="步骤1: 关闭加热\n步骤2: 松开模具",
                title="换模作业指导书",
                equipment_model="海天MA1200"
            )
            for c in chunks:
                print(f"[{c.chunk_type}] {c.chunk_id}: {c.content[:50]}...")
            # 输出:
            # [sop_step] doc_000_chunk_0: 步骤1: 关闭加热
            # [sop_step] doc_000_chunk_1: 步骤2: 松开模具
        """
        content_type = self._detect_content_type(text)

        # 按类型选择分块策略
        if content_type in ("table", "alarm_table"):
            contents = self._chunk_table(text)
        elif content_type == "sop_step":
            contents = self._chunk_by_steps(text)
        else:
            contents = self._chunk_semantic(text)

        # 构建 Chunk 对象
        chunks = []
        for i, content in enumerate(contents):
            if not content.strip():
                continue
            chunk_id = f"{title.replace(' ', '_')[:30]}_chunk_{i}"
            chunks.append(Chunk(
                chunk_id=chunk_id,
                content=content,
                chunk_index=i,
                chunk_type=content_type,
                heading=heading,
                doc_title=title,
                metadata={"equipment_model": equipment_model} if equipment_model else {},
            ))

        return chunks
