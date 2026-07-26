"""爬取功能单元测试（离线，不访问真实站点）。

覆盖:
  - 类型映射与未支持类型校验
  - 文件落地格式（首行为标题）与 manifest 增量记录往返
  - FastAPI 路由: /api/crawl/types 与 /api/crawl 任务提交 + 状态轮询
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.knowledge.crawler import NpcLawCrawler, TYPE_MAP
from src.knowledge.crawler.npc_crawler import CrawlResult


# ---------------------------------------------------------------------------
# 1. 类型映射 / 校验
# ---------------------------------------------------------------------------

def test_type_map_keys():
    assert set(TYPE_MAP.keys()) >= {
        "constitution", "law", "regulation", "supervision", "judicial", "local"
    }
    # 每个类型都有 (flfg_code_ids, subdir) 二元组
    for codes, subdir in TYPE_MAP.values():
        assert isinstance(codes, (list, tuple)) and codes
        assert all(isinstance(c, int) for c in codes)
        assert isinstance(subdir, str) and subdir


def test_unsupported_case_raises():
    crawler = NpcLawCrawler(law_data_dir=Path("/tmp/__noop__"))
    with pytest.raises(ValueError):
        crawler.crawl(doc_type="case")


# ---------------------------------------------------------------------------
# 2. 落地格式 + manifest 往返（临时目录，不联网）
# ---------------------------------------------------------------------------

def test_save_format_and_manifest_roundtrip(tmp_path: Path):
    crawler = NpcLawCrawler(law_data_dir=tmp_path)

    out_dir = tmp_path / "laws"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 写一条
    rel = crawler._save(out_dir, "id001", "中华人民共和国刑法", "第一条 为了惩罚犯罪……", "2024-01-01")
    file_path = tmp_path / rel
    assert file_path.exists()
    text = file_path.read_text(encoding="utf-8")
    assert text.startswith("中华人民共和国刑法\n\n"), "首行应为法律标题"
    assert "第一条" in text

    # manifest 写入后可读回一致
    crawler._save_manifest(out_dir, {"id001": {"id": "id001", "title": "中华人民共和国刑法"}})
    loaded = crawler._load_manifest(out_dir)
    assert loaded["id001"]["title"] == "中华人民共和国刑法"

    # 文件名非法字符被清洗（如含 / 或 : 的标题）
    rel2 = crawler._save(out_dir, "id002", '测试/法:规"', "内容", "")
    assert "/" not in rel2 and ":" not in rel2


# ---------------------------------------------------------------------------
# 3. API 路由（最小 app，避免触发主 lifespan 加载引擎）
# ---------------------------------------------------------------------------

def _make_client() -> TestClient:
    app = FastAPI()
    from src.api.routes import router as api_router
    app.include_router(api_router, prefix="/api")
    return TestClient(app)


def test_crawl_types_endpoint():
    client = _make_client()
    resp = client.get("/api/crawl/types")
    assert resp.status_code == 200
    data = resp.json()
    assert data["source"] == "npc"
    assert "law" in data["types"]
    # unsupported 是字符串列表，校验其中条目提及 case
    assert any("case" in s for s in data["unsupported"])


def test_crawl_submit_and_status(monkeypatch):
    """用假爬虫替换真实爬虫，验证任务提交→状态轮询→结果流程（离线）。"""

    class FakeCrawler:
        def crawl(self, **kwargs):
            # 校验参数确实传到了爬虫
            assert kwargs["doc_type"] == "law"
            return CrawlResult(total=1, added=1, updated=0, skipped=0, failed=0,
                               files=["laws/test.txt"], errors=[])

    monkeypatch.setattr("src.knowledge.crawler.NpcLawCrawler", FakeCrawler)

    client = _make_client()
    resp = client.post("/api/crawl", json={"doc_type": "law", "limit": 1})
    assert resp.status_code == 200
    body = resp.json()
    task_id = body["task_id"]
    assert body["status"] == "pending"

    # 轮询直到完成
    status = None
    for _ in range(50):
        status = client.get(f"/api/crawl/status/{task_id}").json()
        if status["finished"]:
            break
        time.sleep(0.1)

    assert status is not None and status["finished"], "爬取任务应在超时前完成"
    assert status["status"] == "done"
    assert status["result"]["added"] == 1
    assert status["progress"]["total"] == 1


def test_crawl_status_404():
    client = _make_client()
    resp = client.get("/api/crawl/status/nonexistent")
    assert resp.status_code == 404
