"""快速查看法条内容，辅助构建评估数据集"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunking.parser import LawParser

checks = [
    ("治安管理处罚法(2025修订)", [10, 16, 12, 20]),
    ("民法典", [17, 18, 19, 20, 21, 22]),
    ("专利法(2020修正)", [42, 22]),
    ("刑法(2023修正)", [20, 21]),
    ("商标法(2019修正)", [63, 64, 57]),
    ("公司法(2023修订)", [67, 1]),
    ("行政处罚法(2021修订)", [9, 8]),
    ("行政复议法(2023修订)", [20, 21, 30]),
    ("劳动法(2018修正)", [25, 26, 39]),
    ("反不正当竞争法(2025修订)", [2, 6]),
    ("证券法(2019修订)", [1, 55]),
    ("著作权法(2020修正)", [10, 24]),
    ("食品安全法(2021修正)", [4, 148]),
    ("环境保护法(2014修订)", [6, 42]),
    ("道路交通安全法(2021修正)", [19, 91]),
]

p = LawParser()
for law_name, nums in checks:
    for fp in Path("LawData").glob(f"*{law_name}*"):
        doc = p.parse_file(fp)
        print(f"\n{'='*60}")
        print(f"{doc.title}  (共{len(doc.articles)}条)")
        for art in doc.articles:
            if art.number_int in nums:
                print(f"\n  第{art.number}条")
                print(f"  {art.content[:150]}...")
        break
