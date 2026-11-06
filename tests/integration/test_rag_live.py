"""Opt-in isolated end-to-end RAG: real PDF, DB, Ollama, DeepSeek and HTTP."""
import json
import os
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from app.config import get_settings
from app.main import app
from tests.integration.test_rag_postgres import cleanup,initialize_database


pytestmark=[pytest.mark.postgres,pytest.mark.skipif(os.environ.get("RUN_RAG_LIVE_TESTS")!="1",
    reason="Explicit RUN_RAG_LIVE_TESTS=1 with model key and test database is required")]


def test_real_cortex_upload_questions_restart_and_insufficiency(monkeypatch,tmp_path):
    from app.config import Settings
    if Settings().deepseek_api_key is None:
        pytest.skip("Configure a local DEEPSEEK_API_KEY")
    database=os.environ["TEST_PGVECTOR_DATABASE_URL"]
    assert database.rsplit("/",1)[-1].endswith("_test"),"Use a dedicated test database"
    source=Path(os.environ["PAPER_PDF_PATH"])
    initialize_database()
    monkeypatch.setenv("DATABASE_URL",database)
    monkeypatch.setenv("RAG_ENABLED","true")
    monkeypatch.setenv("RAG_STORAGE_DIR",str(tmp_path/"uploads"))
    get_settings.cache_clear()
    papers,traces=[],[]
    report={"scope":"Isolated real end-to-end HTTP; original application database unchanged","questions":[]}

    def ask(client,paper_id,question):
        response=client.post(f"/papers/{paper_id}/questions",json={"question":question,"top_k":5})
        assert response.status_code==200,response.text
        answer=response.json()
        traces.append(answer["trace_id"])
        assert answer["paper_id"]==paper_id
        assert answer["citations"] or answer["insufficient_evidence"]
        record=client.get("/traces/"+answer["trace_id"]).json()
        assert record["status"]=="ok" and len(record["spans"])==6
        report["questions"].append({"question":question,"result":answer,"trace":record})
        return answer

    try:
        with TestClient(app) as client:
            with source.open("rb") as stream:
                response=client.post("/papers",files={"file":(source.name,stream,"application/pdf")},
                                     data={"title":"Cortex live acceptance"})
            assert response.status_code==201,response.text
            imported=response.json()
            papers.append(imported["paper_id"])
            traces.append(imported["trace_id"])
            assert imported["page_count"]==16 and imported["chunk_count"]==101
            trace=client.get("/traces/"+imported["trace_id"]).json()
            assert trace["status"]=="ok" and len(trace["spans"])==7
            report["import"]={"result":imported,"trace":trace}
            ask(client,imported["paper_id"],"Cortex在实验中使用了哪些比较基线？请保留基线名称及适用条件。")
        with TestClient(app) as restarted:
            assert restarted.get("/traces/"+imported["trace_id"]).status_code==200
            ask(restarted,imported["paper_id"],"论文的实验使用了什么模型和数据集？请保留名称，并说明证据没有覆盖的部分。")
            absent=ask(restarted,imported["paper_id"],"论文是否报告了在火星上运行Cortex的实验结果？")
            assert absent["insufficient_evidence"] is True
        report["checks"]={"pages":16,"chunks":101,"import_spans":7,"question_spans":6,
                          "new_lifespan_reused_persistent_paper":True,"missing_evidence_reported":True}
        if os.environ.get("RAG_LIVE_REPORT_PATH"):
            Path(os.environ["RAG_LIVE_REPORT_PATH"]).write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(report["checks"]))
    finally:
        get_settings.cache_clear()
        cleanup(papers,traces)
