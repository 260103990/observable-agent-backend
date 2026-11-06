"""Explicit live full Search-Agent no-cache/cold/priming/warm comparison."""
import argparse,asyncio,hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.config import Settings
from app.evaluation.models import Workload
from app.evaluation.agent_cache_benchmark import compare_agent_cache


async def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workload',type=Path,default=ROOT/'docs/evaluation/v3_agent_cases.json')
    parser.add_argument('--repeats',type=int,choices=range(1,6),default=2)
    parser.add_argument('--paper-id',default=os.getenv('EVAL_PAPER_ID'))
    args=parser.parse_args()
    if not args.live:parser.error('--live is required for paid model and Search calls')
    content=args.workload.read_bytes()
    workload=Workload.model_validate_json(content,strict=True)
    report=await compare_agent_cache(settings=Settings(),workload=workload,repeats=args.repeats,output_dir=args.output,
        paper_id=args.paper_id,database_url=os.getenv('EVAL_DATABASE_URL'),verification_tier='live',
        workload_file_sha256=hashlib.sha256(content).hexdigest())
    print(json.dumps({'phases':report['phases'],'comparison':report['comparison'],'manual_review':report['manual_review']},ensure_ascii=True))


if __name__=='__main__':
    try:asyncio.run(main())
    except Exception as error:
        print('Agent cache comparison failed: '+type(error).__name__+'. Check explicit configuration and fresh output directory.',file=sys.stderr)
        raise SystemExit(2) from None
