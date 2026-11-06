"""Explicit live Search-tool no-cache/cold/primed-warm experiment, never connects to a database."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import httpx
from app.config import Settings
from app.evaluation.search_cache_benchmark import SearchBenchmarkWorkload,evaluate_search_cache


async def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workload',type=Path,default=ROOT/'docs/evaluation/v3_search_cases.json')
    parser.add_argument('--repeats',type=int,choices=range(1,6),default=2)
    args=parser.parse_args()
    if not args.live:parser.error('--live is required for paid Search requests')
    content=args.workload.read_bytes()
    workload=SearchBenchmarkWorkload.model_validate_json(content,strict=True)
    settings=Settings()
    async with httpx.AsyncClient(trust_env=False) as http:
        report=await evaluate_search_cache(client=http,settings=settings,workload=workload,repeats=args.repeats,
            output_dir=args.output,verification_tier='live',workload_file_sha256=hashlib.sha256(content).hexdigest())
    print(json.dumps({'run_id':report['run_id'],'measurement_scope':report['measurement_scope'],
        'phases':report['phases'],'manual_review':report['manual_review']},ensure_ascii=True))


if __name__=='__main__':
    try:asyncio.run(main())
    except Exception as error:
        print('Search cache evaluation failed: '+type(error).__name__+'. Check local configuration and fresh output directory.',file=sys.stderr)
        raise SystemExit(2) from None
