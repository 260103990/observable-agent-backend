"""Explicit live cold Search concurrency comparison; never connects to a database."""
import argparse,asyncio,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import httpx
from app.config import Settings
from app.evaluation.search_coalescing_benchmark import evaluate_search_coalescing


async def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--query',default='Qwen3-0.6B Hugging Face official model card')
    parser.add_argument('--count',type=int,choices=range(1,11),default=5)
    parser.add_argument('--callers',type=int,choices=range(2,17),default=4)
    parser.add_argument('--repeats',type=int,choices=range(1,6),default=2)
    args=parser.parse_args()
    if not args.live:parser.error('--live is required for paid external requests')
    report=await evaluate_search_coalescing(client_factory=lambda:httpx.AsyncClient(trust_env=False),settings=Settings(),
        query=args.query,count=args.count,callers=args.callers,repeats=args.repeats,output_dir=args.output,verification_tier='live')
    print(json.dumps({'run_id':report['run_id'],'modes':report['modes'],
        'control_valid':all(b['cold_overlap_control_valid'] for b in report['bursts'] if b['mode']=='disabled'),
        'coalescing_exercised':all(b['coalescing_exercised'] for b in report['bursts'] if b['mode']=='enabled'),
        'manual_review':report['manual_review']},ensure_ascii=True))


if __name__=='__main__':
    try:asyncio.run(main())
    except Exception as error:
        print('Search coalescing evaluation failed: '+type(error).__name__+'. Check local configuration and fresh output directory.',file=sys.stderr)
        raise SystemExit(2) from None
