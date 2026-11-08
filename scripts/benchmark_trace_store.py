"""Explicit local database benchmark; never uses application DATABASE_URL or model APIs."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.evaluation.trace_load import benchmark_trace_store


async def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--operations',type=int,default=100)
    parser.add_argument('--repeats',type=int,default=2)
    parser.add_argument('--spans',type=int,default=4)
    parser.add_argument('--timeout',type=float,default=5)
    parser.add_argument('--order-seed',type=int,default=20261107)
    args=parser.parse_args()
    if not args.live:parser.error('--live is required to create an isolated benchmark database')
    report=await benchmark_trace_store(os.environ.get('TRACE_BENCH_DATABASE_URL'),args.output,
        operations=args.operations,repeats=args.repeats,span_count=args.spans,
        timeout=args.timeout,order_seed=args.order_seed)
    print(json.dumps({'success':report['success'],'phases':len(report['phases']),'cleanup':report['cleanup']}))
    return 0 if report['success'] else 1


if __name__=='__main__':
    try:raise SystemExit(asyncio.run(main()))
    except Exception as error:
        print('Trace benchmark failed: '+type(error).__name__+'. Inspect local configuration/output.',file=sys.stderr)
        raise SystemExit(2) from None
