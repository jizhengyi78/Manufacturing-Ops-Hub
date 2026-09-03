"""
Milvus 向量数据库浏览器
=======================
在 PyCharm 里直接 Run 这个脚本，交互式浏览向量库数据。

用法: python scripts/browse_milvus.py
"""

from pymilvus import MilvusClient
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "milvus_lite.db"


def browse():
    client = MilvusClient(str(DB_PATH))

    print("=" * 70)
    print("  Milvus 向量数据库浏览器")
    print("=" * 70)

    cols = client.list_collections()
    if not cols:
        print("\n⚠ 数据库为空，请先执行 /admin/seed 加载种子数据")
        return

    # 选择一个 Collection
    print(f"\n可用的 Collection ({len(cols)}):")
    for i, c in enumerate(cols):
        stats = client.get_collection_stats(c)
        print(f"  [{i+1}] {c} — {stats['row_count']} 条")

    choice = input(f"\n选 Collection (1-{len(cols)}, 默认1): ").strip()
    idx = int(choice) - 1 if choice else 0
    col = cols[idx]

    client.load_collection(col)
    stats = client.get_collection_stats(col)
    total = stats["row_count"]

    page_size = 10
    page = 0
    max_page = (total - 1) // page_size

    while True:
        offset = page * page_size
        results = client.query(
            collection_name=col,
            filter="id != \"\"",
            output_fields=["id", "content", "workshop_id", "source", "doc_title", "equipment_model"],
            limit=page_size,
            offset=offset,
        )

        print(f"\n{'=' * 70}")
        print(f"  {col}  (共 {total} 条)  第 {page + 1}/{max_page + 1} 页")
        print(f"{'=' * 70}")

        for i, r in enumerate(results):
            cid = r.get("id", "?")[:40]
            title = r.get("doc_title", "?")[:30]
            ws = r.get("workshop_id", "?")
            eq = r.get("equipment_model", "")
            content = r.get("content", "")[:120].replace("\n", " ")

            print(f"\n  [{offset + i + 1}] {title}")
            print(f"      id={cid}  workshop={ws}  equipment={eq}")
            print(f"      {content}...")

        print(f"\n  [N]下一页  [P]上一页  [Q]退出  [G]跳转到第N页")

        cmd = input("  > ").strip().lower()
        if cmd == "q":
            break
        elif cmd == "n" and page < max_page:
            page += 1
        elif cmd == "p" and page > 0:
            page -= 1
        elif cmd.startswith("g"):
            try:
                n = int(cmd[1:])
                if 1 <= n <= max_page + 1:
                    page = n - 1
            except ValueError:
                pass

    client.release_collection(col)
    print("\n再见 👋")


if __name__ == "__main__":
    browse()
