"""Run an explicitly authorized fixed workload; never migrate or write paper data."""
import argparse
import asyncio
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from app.config import Settings
from app.evaluation.models import Workload
from app.evaluation import runner as evaluation_module
from app.evaluation.runner import evaluate_cases
from app.evaluation.runtime import evaluation_session


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live',action='store_true',help='Explicitly permit paid external model and Search requests')
    parser.add_argument('--workload',type=Path,default=ROOT/'docs/evaluation/v2_cases.json')
    parser.add_argument('--output',type=Path,required=True,help='Fresh output directory; existing reports are never replaced')
    parser.add_argument('--paper-id',default=os.getenv('EVAL_PAPER_ID'))
    parser.add_argument('--case',action='append',dest='cases',help='Run a named case; repeat to select multiple cases')
    parser.add_argument('--repeat',type=int,choices=range(1,6),default=1)
    args=parser.parse_args()
    if not args.live:
        parser.error('Pass --live to explicitly permit external service calls')
    return args


async def run(args):
    content=args.workload.read_bytes()
    workload=Workload.model_validate_json(content,strict=True)
    if args.cases:
        unknown=set(args.cases)-{case.id for case in workload.cases}
        if unknown:
            raise ValueError('Unknown evaluation case ID')
        workload=workload.model_copy(update={'cases':[case for case in workload.cases if case.id in set(args.cases)]})
    settings=Settings()
    async with evaluation_session(settings=settings,workload=workload,paper_id=args.paper_id,
                                  database_url=os.getenv('EVAL_DATABASE_URL')) as session:
        from app import main
        source_root=Path(main.__file__).parents[1]
        source_paths=['app/answer_policy.py','app/agent/loop_runner.py','app/agent/loop_models.py','app/agent/deepseek_loop.py',
            'app/agent/paper_tool.py','app/agent/web_evidence.py','app/tools/web_search.py',
            'app/api/loop_dependencies.py','app/tools/search_cache.py','app/tools/search_coalescing.py','app/rag/service.py','app/rag/store.py',
            'app/agent/final_output.py','app/agent/evidence_history.py','app/rag/hybrid.py','app/tools/web_models.py','app/config.py']
        session.manifest.update(workload_file_sha256=hashlib.sha256(content).hexdigest(),
            source_sha256={rel:hashlib.sha256((source_root/rel).read_bytes()).hexdigest() for rel in source_paths},
            evaluation_source_sha256={name:hashlib.sha256((Path(evaluation_module.__file__).parent/name).read_bytes()).hexdigest()
                for name in ('models.py','recording.py','scoring.py','runner.py','runtime.py')},
            evaluation_cli_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            packages={name:version(name) for name in ('fastapi','httpx','asyncpg','pydantic')},
            python_version=sys.version.split()[0])
        result=await evaluate_cases(client=session.client,workload=workload,recorder=session.recorder,
            capture=session.capture,paper_id=args.paper_id,output_dir=args.output,
            manifest=session.manifest,repeats=args.repeat)
    print(json.dumps({'run_id':result['run_id'],'case_runs':result['case_runs'],
                      'automatic_checks_passed':result['automated_pass_count'],
                      'manual_review_pending':result['manual_review_pending'],'output':str(args.output)},ensure_ascii=True))


if __name__=='__main__':
    try:
        asyncio.run(run(parse_args()))
    except Exception as error:
        # Do not print Settings, DSNs, remote response bodies or chained credential-bearing errors.
        print('Evaluation failed: '+type(error).__name__+'. Check explicit arguments, source hash and local service setup.',file=sys.stderr)
        raise SystemExit(2) from None
