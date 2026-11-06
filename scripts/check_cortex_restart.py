"""Isolated, owned uvicorn process restart; no migration or original app switch."""
import asyncio
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time

import asyncpg
import httpx

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tests.integration.test_rag_postgres import initialize_database,cleanup

PORT=18081
BASE=f"http://127.0.0.1:{PORT}"
RUN=Path(tempfile.mkdtemp(prefix="rag-process-acceptance-"))
SOURCE=Path(os.environ["PAPER_PDF_PATH"])
REPORT=Path(os.environ.get("RAG_PROCESS_REPORT_PATH",str(RUN/"acceptance.json")))
assert os.environ["TEST_PGVECTOR_DATABASE_URL"].rsplit("/",1)[-1].endswith("_test")
with socket.socket() as probe:
    probe.bind(("127.0.0.1",PORT))
initialize_database()
environment=os.environ.copy()
environment.update(DATABASE_URL=os.environ["TEST_PGVECTOR_DATABASE_URL"],RAG_ENABLED="true",
                   RAG_STORAGE_DIR=str(RUN/"papers"),PYTHONDONTWRITEBYTECODE="1",PYTHONIOENCODING="utf-8")
papers,traces,processes=[],[],[]
logs=[]

def start():
    log=(RUN/f"server-{len(processes)+1}.log").open("wb")
    logs.append(log)
    process=subprocess.Popen([sys.executable,"-B","-m","uvicorn","app.main:app","--host","127.0.0.1",
        "--port",str(PORT),"--limit-concurrency","20"],cwd=str(ROOT),env=environment,stdout=log,stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
    processes.append(process)
    until=time.monotonic()+30
    while time.monotonic()<until:
        if process.poll() is not None:
            raise RuntimeError("Owned test server failed to start; inspect the local acceptance log")
        try:
            if httpx.get(BASE+"/health",timeout=1,trust_env=False).status_code==200:
                return process
        except httpx.RequestError:
            pass
        time.sleep(0.2)
    raise RuntimeError("Owned server startup timed out")

def stop(process):
    if process.poll() is None:
        process.terminate()
        process.wait(timeout=10)

try:
    first=start()
    with httpx.Client(base_url=BASE,timeout=120,trust_env=False) as client:
        with SOURCE.open("rb") as stream:
            response=client.post("/papers",files={"file":(SOURCE.name,stream,"application/pdf")},
                                 data={"title":"Cortex process restart acceptance"})
        assert response.status_code==201,response.text
        imported=response.json()
        papers.append(imported["paper_id"])
        traces.append(imported["trace_id"])
        assert (imported["page_count"],imported["chunk_count"])==(16,101)
        assert client.get("/traces/"+imported["trace_id"]).status_code==200
    stop(first)
    second=start()
    assert first.pid!=second.pid
    with httpx.Client(base_url=BASE,timeout=120,trust_env=False) as client:
        old=client.get("/traces/"+imported["trace_id"])
        assert old.status_code==200 and old.json()["status"]=="ok"
        answered=client.post(f"/papers/{imported['paper_id']}/questions",json={
            "question":"实验的搜索与代码生成场景分别使用哪些模型？请只根据证据回答。"})
        assert answered.status_code==200,answered.text
        result=answered.json()
        traces.append(result["trace_id"])
        assert result["citations"] or result["insufficient_evidence"]
        trace=client.get("/traces/"+result["trace_id"]).json()
        assert trace["status"]=="ok" and len(trace["spans"])==6
        report={"scope":"Real TCP HTTP and two independent uvicorn processes, dedicated test DB",
                "first_pid":first.pid,"second_pid":second.pid,"import":imported,"answer":result,"trace":trace,
                "old_import_trace_readable":True}
        REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        print(json.dumps({"different_processes":True,"old_import_trace_readable":True,
                          "pages":16,"chunks":101,"question_spans":6,"citation_count":len(result["citations"])}))
finally:
    for process in processes:
        stop(process)
    for log in logs:
        log.close()
    cleanup(papers,traces)
