"""Explicit live collection smoke, not an assertion that every answer is correct."""
import asyncio
import os
from pathlib import Path

import pytest

from app.config import Settings
from app.evaluation.models import Workload
from app.evaluation.runner import evaluate_cases
from app.evaluation.runtime import evaluation_session
from app import main


pytestmark=pytest.mark.postgres


def test_live_evaluation_collects_six_cases_and_keeps_quality_review_separate(tmp_path):
    if os.getenv('RUN_AGENT_EVAL_LIVE_TESTS')!='1':
        pytest.skip('Opt in explicitly with RUN_AGENT_EVAL_LIVE_TESTS=1')
    url,paper_id=os.getenv('EVAL_DATABASE_URL'),os.getenv('EVAL_PAPER_ID')
    if not url or not paper_id or Settings().deepseek_api_key is None:
        pytest.skip('Configure explicit readonly paper connection, paper ID and local key')
    root=Path(main.__file__).parents[1]
    workload=Workload.model_validate_json((root/'docs/evaluation/v2_cases.json').read_bytes(),strict=True)
    async def run():
        async with evaluation_session(settings=Settings(),workload=workload,paper_id=paper_id,database_url=url) as session:
            return await evaluate_cases(client=session.client,workload=workload,recorder=session.recorder,
                capture=session.capture,paper_id=paper_id,output_dir=tmp_path,manifest=session.manifest)
    report=asyncio.run(run())
    assert report['case_runs']==6 and report['manual_review_pending']==6
    assert all(row['http_status'] is not None for row in report['results'])
    assert report['manifest']['database_mode']=='read_only_existing_paper'
    assert (tmp_path/'cases.jsonl').is_file() and (tmp_path/'report.md').is_file()
