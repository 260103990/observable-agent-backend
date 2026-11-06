"""Fixed Search-tool workload; deliberately separate from Agent/answer-quality evaluation."""
from collections import Counter
from datetime import datetime,timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel,ConfigDict,Field,model_validator

from app.config import Settings
from app.evaluation.recording import EvaluationRecorder
from app.tools import search_cache,search_coalescing,web_models,web_search
from app.tools.search_cache import cache_from_settings
from app.tools.web_models import SearchInput
from app.tools.web_search import DeepSeekSearchClient


class SearchBenchmarkCase(SearchInput):
    id:str=Field(pattern=r'^[a-z][a-z0-9_]{0,63}$')


class SearchBenchmarkWorkload(BaseModel):
    model_config=ConfigDict(strict=True,extra='forbid')
    version:str=Field(min_length=1,max_length=128)
    cases:list[SearchBenchmarkCase]=Field(min_length=1,max_length=10)

    @model_validator(mode='after')
    def unique_cases(self):
        if len({c.id for c in self.cases})!=len(self.cases):raise ValueError('Duplicate case ID')
        if len({(c.query,c.count) for c in self.cases})!=len(self.cases):
            raise ValueError('Duplicate normalized Search parameter profile')
        return self


def _fingerprint(results):
    content=json.dumps(results,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')
    return hashlib.sha256(content).hexdigest()


async def evaluate_search_cache(*,client,settings:Settings,workload:SearchBenchmarkWorkload,
        repeats:int,output_dir:Path,verification_tier:Literal['controlled','live']='controlled',
        workload_file_sha256=None):
    if type(repeats) is not int or not 1<=repeats<=5:raise ValueError('Use 1 to 5 repeats')
    if settings.deepseek_api_key is None:raise ValueError('Search benchmark needs a configured key')
    output_dir=Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=False)
    cache=cache_from_settings(settings)
    recorder=EvaluationRecorder()
    run_id=str(uuid4())
    manifest={'run_id':run_id,'started_at':datetime.now(timezone.utc).isoformat(),
        'verification_tier':verification_tier,'measurement_scope':'sequential_search_tool_not_agent',
        'workload':workload.model_dump(),'workload_file_sha256':workload_file_sha256,
        'canonical_workload_sha256':_fingerprint(workload.model_dump()),'repeats':repeats,
        'provider':'deepseek','endpoint':web_search.SEARCH_URL,'need_content':False,'localization':'',
        'cache_backend':'process_local_memory','cache_config':{name:getattr(settings,'web_search_cache_'+name)
            for name in ('ttl_seconds','max_entries','max_bytes','max_entry_bytes','timeout_seconds',
                         'coalesce_enabled','max_inflight','max_waiters')},
        'search_config':{'timeout_seconds':settings.web_search_timeout_seconds,
                         'max_response_bytes':settings.web_search_max_response_bytes},
        'source_sha256':{name:hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            for name,module in (('web_search',web_search),('search_cache',search_cache),
                                ('search_coalescing',search_coalescing),('web_models',web_models))},
        'benchmark_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'packages':{name:version(name) for name in ('httpx','pydantic')},
        'phase_order':['bypass','cold','priming','warm'],
        'limitations':['Fixed sequential phase order and small sample; no causal/stability claim.',
            'Response body bytes exclude requests, headers and TLS.',
            'No Agent/LLM latency, semantic correctness or freshness benefit is measured.',
            'Priming cost is reported separately, never hidden in warm results.']}
    (output_dir/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    rows=[]
    primed={}
    try:
        with (output_dir/'cases.jsonl').open('x',encoding='utf-8') as stream:
            for phase in ('bypass','cold','priming','warm'):
                if phase=='priming':cache.clear()
                for repeat in range(1,(1 if phase=='priming' else repeats)+1):
                    if phase=='cold':cache.clear()
                    for case in workload.cases:
                        trace_id=str(uuid4())
                        request_id=f'{run_id}-{phase}-{repeat}-{case.id}'
                        search=DeepSeekSearchClient(client=client,api_key=settings.deepseek_api_key,trace_store=recorder,
                            cache=None if phase=='bypass' else cache,cache_timeout_seconds=settings.web_search_cache_timeout_seconds,
                            timeout_seconds=settings.web_search_timeout_seconds,max_response_bytes=settings.web_search_max_response_bytes)
                        started=perf_counter()
                        found,error=None,None
                        try:
                            found=await search.search(case.query,count=case.count,request_id=request_id,trace_id=trace_id)
                        except Exception as exc:
                            error=type(exc).__name__
                        elapsed=(perf_counter()-started)*1000
                        trace=await recorder.get(trace_id)
                        attrs=trace.spans[0].attributes if trace and trace.spans else {}
                        results=[item.model_dump() for item in found.results] if found else []
                        payload_hash=_fingerprint(results) if found else None
                        source=primed.get(case.id)
                        row={'phase':phase,'repeat':repeat,'case_id':case.id,'trace_id':trace_id,'request_id':request_id,
                            'parameters':{'query':case.query,'count':case.count,'need_content':False,'localization':''},
                            'error_type':error,'wall_ms':elapsed,'cache_status':attrs.get('cache_status','unknown'),
                            'cache_error_type':attrs.get('cache_error_type'),
                            'origin_search_trace_id':attrs.get('origin_search_trace_id'),
                            'cache_age_seconds':attrs.get('cache_age_seconds'),
                            'search_request_attempts':attrs.get('outbound_request_attempts',0),
                            'response_body_bytes':attrs.get('response_body_bytes',0),
                            'retrieved_at':found.retrieved_at if found else None,
                            'payload_sha256':payload_hash,'results':results,
                            'warm_payload_matches_primed':bool(source and payload_hash==source['payload_sha256']
                                and found and found.retrieved_at==source['retrieved_at']) if phase=='warm' else None,
                            'trace':trace.model_dump() if trace else None}
                        rows.append(row)
                        if phase=='priming' and found and found.results:primed[case.id]=row
                        stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
                        stream.flush()
    finally:
        cache.clear()
    summaries={}
    for phase in ('bypass','cold','priming','warm'):
        subset=[row for row in rows if row['phase']==phase]
        status=Counter(row['cache_status'] for row in subset)
        successful=[r['wall_ms'] for r in subset if r['error_type'] is None]
        summaries[phase]={'requests':len(subset),'successes':len(successful),'failures':len(subset)-len(successful),
            'cache_hits':status['hit'],'cache_misses':status['miss'],'cache_bypasses':status['bypass'],
            'cache_errors':sum(r['cache_error_type'] is not None for r in subset),
            'search_request_attempts':sum(r['search_request_attempts'] for r in subset),
            'response_body_bytes':sum(r['response_body_bytes'] for r in subset),
            'successful_wall_ms_median':median(successful) if successful else None,
            'successful_wall_ms_min':min(successful) if successful else None,
            'successful_wall_ms_max':max(successful) if successful else None,
            'automatic_reuse_pass':all(r['error_type'] is None and r['cache_status']=='hit'
                and r['search_request_attempts']==0 and r['response_body_bytes']==0
                and r['warm_payload_matches_primed'] is True for r in subset) if phase=='warm' else None}
    report={'run_id':run_id,'verification_tier':verification_tier,'measurement_scope':manifest['measurement_scope'],
        'phases':summaries,'manual_review':{'source_quality':'pending','information_freshness':'pending',
            'answer_quality':'not_measured_no_answer_generation'},'limitations':manifest['limitations']}
    (output_dir/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# Search cache fixed-workload experiment','',f'Run: {run_id}',
        '','| Phase | Requests | Failures | Hits | Outbound attempts | Body bytes | Median ms (successful) |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for phase,item in summaries.items():
        measured=item['successful_wall_ms_median']
        lines.append(f"| {phase} | {item['requests']} | {item['failures']} | {item['cache_hits']} | "
            f"{item['search_request_attempts']} | {item['response_body_bytes']} | {round(measured,3) if measured is not None else '-'} |")
    lines+=['','Source quality/freshness review: pending. Answer quality: not measured.','']
    lines+=['- '+note for note in manifest['limitations']]
    (output_dir/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return report
