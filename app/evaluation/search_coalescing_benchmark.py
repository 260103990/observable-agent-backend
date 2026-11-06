"""Cold concurrent Search bursts, with independent per-request HTTP clients and explicit controls."""
import asyncio
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
from statistics import median
from time import perf_counter
from uuid import uuid4

from app.evaluation.recording import EvaluationRecorder
from app.tools import search_cache,search_coalescing,web_models,web_search
from app.tools.search_cache import cache_from_settings
from app.tools.web_models import SearchInput
from app.tools.web_search import DeepSeekSearchClient


def _payload_hash(results):
    return hashlib.sha256(json.dumps(results,sort_keys=True,ensure_ascii=False,
        separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest()


async def evaluate_search_coalescing(*,client_factory,settings,query,count,callers,repeats,output_dir,
                                    verification_tier='controlled'):
    arguments=SearchInput(query=query,count=count)
    if type(callers) is not int or not 2<=callers<=16:raise ValueError('Use 2 to 16 callers')
    if type(repeats) is not int or not 1<=repeats<=5:raise ValueError('Use 1 to 5 repeats')
    if settings.deepseek_api_key is None:raise ValueError('A configured key is required')
    output_dir=Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=False)
    recorder=EvaluationRecorder();run_id=str(uuid4());bursts=[]
    manifest={'run_id':run_id,'started_at':datetime.now(timezone.utc).isoformat(),
        'verification_tier':verification_tier,'measurement_scope':'concurrent_search_tool_not_agent',
        'parameters':{**arguments.model_dump(),'need_content':False,'localization':''},
        'callers':callers,'repeats':repeats,'mode_order':['disabled','enabled'],
        'cache_backend':'process_local_memory','http_client_ownership':'one_per_logical_request',
        'config':{name:getattr(settings,'web_search_cache_'+name) for name in ('ttl_seconds','max_entries',
            'max_bytes','max_entry_bytes','timeout_seconds','max_inflight','max_waiters')},
        'search_timeout_seconds':settings.web_search_timeout_seconds,
        'max_response_bytes':settings.web_search_max_response_bytes,
        'source_sha256':{name:hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            for name,module in (('search_cache',search_cache),('search_coalescing',search_coalescing),
                                ('web_models',web_models),('web_search',web_search))},
        'benchmark_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'limitations':['Tool-level single-process sample, not Agent performance or production throughput.',
            'Each burst starts with clear cache; barrier coordinates start but cannot guarantee network overlap.',
            'Control validity and actual follower reuse are reported, never inferred from equal text alone.',
            'Body bytes exclude requests, HTTP headers and TLS. Source quality/freshness are not evaluated.']}
    (output_dir/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    with (output_dir/'cases.jsonl').open('x',encoding='utf-8') as stream:
        for mode in ('disabled','enabled'):
            config=settings.model_copy(update={'web_search_cache_coalesce_enabled':mode=='enabled'})
            cache=cache_from_settings(config)
            try:
                for repeat in range(1,repeats+1):
                    cache.clear()
                    ready=asyncio.Event();start=asyncio.Event();ready_count=0
                    def arrived():
                        nonlocal ready_count
                        ready_count+=1
                        if ready_count==callers:ready.set()
                    async def worker(index):
                        trace_id=str(uuid4());found=None;error=None;has_arrived=False;started=perf_counter()
                        try:
                            async with client_factory() as http:
                                arrived();has_arrived=True
                                await start.wait()
                                started=perf_counter()
                                search=DeepSeekSearchClient(client=http,api_key=settings.deepseek_api_key,trace_store=recorder,
                                    cache=cache,cache_timeout_seconds=settings.web_search_cache_timeout_seconds,
                                    timeout_seconds=settings.web_search_timeout_seconds,max_response_bytes=settings.web_search_max_response_bytes)
                                found=await search.search(arguments.query,count=arguments.count,
                                    request_id=f'{run_id}-{mode}-{repeat}-{index}',trace_id=trace_id)
                                wall_ms=(perf_counter()-started)*1000
                        except Exception as exc:
                            error=type(exc).__name__;wall_ms=(perf_counter()-started)*1000
                        finally:
                            if not has_arrived:arrived()
                        trace=await recorder.get(trace_id)
                        attrs=trace.spans[0].attributes if trace and trace.spans else {}
                        results=[r.model_dump() for r in found.results] if found is not None else None
                        return {'mode':mode,'repeat':repeat,'caller':index,'trace_id':trace_id,
                            'source_trace_id':attrs.get('origin_search_trace_id',trace_id),'wall_ms':wall_ms,
                            'error_type':error,'cache_status':attrs.get('cache_status','unknown'),
                            'coalescing_role':attrs.get('coalescing_role'),
                            'coalescing_status':attrs.get('coalescing_status'),
                            'search_request_attempts':attrs.get('outbound_request_attempts',0),
                            'response_body_bytes':attrs.get('response_body_bytes',0),
                            'retrieved_at':found.retrieved_at if found is not None else None,
                            'payload_sha256':_payload_hash(results) if results is not None else None,
                            'results':results,'trace':trace.model_dump() if trace else None}
                    tasks=[asyncio.create_task(worker(i)) for i in range(callers)]
                    rows=[]
                    try:
                        await asyncio.wait_for(ready.wait(),10)
                        burst_started=perf_counter();start.set()
                        for completed in asyncio.as_completed(tasks,timeout=settings.web_search_timeout_seconds+15):
                            row=await completed
                            rows.append(row)
                            stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');stream.flush()
                        burst_wall_ms=(perf_counter()-burst_started)*1000
                    finally:
                        for task in tasks:
                            if not task.done():task.cancel()
                        await asyncio.gather(*tasks,return_exceptions=True)
                    attempts=sum(r['search_request_attempts'] for r in rows)
                    reused=sum(r['coalescing_status']=='reused' for r in rows)
                    successes=sum(r['error_type'] is None for r in rows)
                    owners=[r for r in rows if r['coalescing_role']=='leader' and r['search_request_attempts']==1]
                    shared_valid=(len(owners)==1 and all(r['source_trace_id']==owners[0]['trace_id'] for r in rows)
                        and len({r['payload_sha256'] for r in rows})==1 and len({r['retrieved_at'] for r in rows})==1
                        and all(r['search_request_attempts']==0 and r['response_body_bytes']==0
                            for r in rows if r['coalescing_role']=='follower'))
                    bursts.append({'mode':mode,'repeat':repeat,'logical_calls':len(rows),'successes':successes,
                        'failures':len(rows)-successes,'search_request_attempts':attempts,
                        'response_body_bytes':sum(r['response_body_bytes'] for r in rows),
                        'coalesced_calls':reused,'cache_hits':sum(r['cache_status']=='hit' for r in rows),
                        'burst_wall_ms':burst_wall_ms,
                        'cold_overlap_control_valid':all(r['cache_status']=='miss' for r in rows) and attempts==callers if mode=='disabled' else None,
                        'coalescing_exercised':mode=='enabled' and successes==callers and reused>0 and attempts==1 and shared_valid})
            finally:cache.clear()
    modes={}
    for mode in ('disabled','enabled'):
        selected=[r for r in bursts if r['mode']==mode]
        modes[mode]={name:sum(r[name] for r in selected) for name in ('logical_calls','successes','failures',
            'search_request_attempts','response_body_bytes','coalesced_calls','cache_hits')}
        modes[mode]['burst_wall_ms_median']=median(r['burst_wall_ms'] for r in selected)
    report={'run_id':run_id,'verification_tier':verification_tier,'measurement_scope':manifest['measurement_scope'],
        'bursts':bursts,'modes':modes,'manual_review':{'source_quality':'pending','information_freshness':'pending',
        'answer_quality':'not_measured'},'limitations':manifest['limitations']}
    (output_dir/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# Concurrent Search coalescing experiment','',f'Run: {run_id}',
        '','| Mode | Logical calls | Failures | Outbound attempts | Body bytes | Coalesced calls | Warm-cache hits |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for mode,r in modes.items():
        lines.append(f"| {mode} | {r['logical_calls']} | {r['failures']} | {r['search_request_attempts']} | "
                     f"{r['response_body_bytes']} | {r['coalesced_calls']} | {r['cache_hits']} |")
    lines+=['','Source quality/freshness pending; no LLM answers generated.','']
    lines+=['- '+item for item in manifest['limitations']]
    (output_dir/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return report
