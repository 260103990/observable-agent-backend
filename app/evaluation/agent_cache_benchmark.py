"""Full Search Agent comparisons through the unchanged ASGI API, not a fixed-model replay."""
from datetime import datetime,timezone
import hashlib
import json
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from statistics import median
import sys

from app.evaluation.runner import evaluate_cases
from app.evaluation.runtime import evaluation_session
from app.tools.web_models import SearchInput


def _canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)


def _observed_web(row):
    profiles=[];outcomes=[];sources={}
    for tool in row['observations'].get('tool_results',[]):
        if tool.get('name')!='web_search':continue
        succeeded=tool.get('result',{}).get('ok') is True
        outcomes.append(succeeded)
        try:
            args=SearchInput.model_validate(tool['arguments'])
            profiles.append({**args.model_dump(),'need_content':False,'localization':''})
        except (ValueError,TypeError):
            profiles.append({'invalid_arguments':True})
        if not succeeded:continue
        for item in tool['result']['output'].get('evidence',[]):
            snapshot={k:v for k,v in item.items() if k not in ('evidence_id','search_trace_id')}
            sources[_canonical(snapshot)]=snapshot
    contents={_canonical({k:v for k,v in snapshot.items() if k!='retrieved_at'}) for snapshot in sources.values()}
    return {'profiles':profiles,'tool_successes':outcomes,'source_snapshots':list(sources.values()),
        'source_snapshot_sha256':hashlib.sha256(_canonical(sorted(sources)).encode('utf-8')).hexdigest() if sources else None,
        'source_content_sha256':hashlib.sha256(_canonical(sorted(contents)).encode('utf-8')).hexdigest() if contents else None}


def _summary(report):
    rows=report['results']
    summary={'case_runs':len(rows),'completed':sum(r['result'].get('status')=='completed' for r in rows),
        'http_failures':sum(r['http_status']!=200 or r['error_type'] is not None for r in rows),
        'automatic_checks_passed':sum(r['automated_pass'] for r in rows),
        'api_wall_ms_total':sum(r['api_wall_ms'] for r in rows),
        'api_wall_ms_median':median(r['api_wall_ms'] for r in rows)}
    for target,source in {'llm_duration_ms':'model_duration_ms','llm_turns':'model_turns','tool_duration_ms':'tool_duration_ms',
        'search_attempts':'search_request_attempts','response_body_bytes':'search_response_body_bytes',
        'cache_hits':'search_cache_hits','cache_misses':'search_cache_misses','cache_bypasses':'search_cache_bypasses',
        'cache_errors':'search_cache_errors','coalesced_calls':'search_coalesced_calls'}.items():
        summary[target]=sum(r['metrics'].get(source,0) for r in rows)
    return summary


async def compare_agent_cache(*,settings,workload,repeats,output_dir,paper_id=None,database_url=None,
        session_factory=evaluation_session,verification_tier='controlled',workload_file_sha256=None):
    if type(repeats) is not int or not 1<=repeats<=5:raise ValueError('Use 1 to 5 repeats')
    if not any(case.enable_web_search for case in workload.cases):raise ValueError('A Search workload is required')
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=False)
    header={'started_at':datetime.now(timezone.utc).isoformat(),'verification_tier':verification_tier,
        'measurement_scope':'full_search_agent_in_process_asgi','workload':workload.model_dump(),
        'workload_file_sha256':workload_file_sha256,'repeats':repeats,'phase_order':['bypass','cold','priming','warm'],
        'query_policy':'Prompts constrain query/count; actual model decisions are recorded and compared.',
        'priming_policy':'Full Agent run, its LLM/Search/API costs reported separately.'}
    modules=('app.config','app.main','app.answer_policy','app.api.loop_dependencies','app.api.routes.agent_loop',
        'app.agent.loop_runner','app.agent.loop_models','app.agent.loop_trace','app.agent.deepseek_loop',
        'app.agent.evidence_history',
        'app.rag.hybrid',
        'app.rag.store','app.rag.service','app.agent.paper_tool',
        'app.agent.final_output','app.agent.web_evidence','app.tools.web_search','app.tools.web_models',
        'app.tools.search_cache','app.tools.search_coalescing','app.evaluation.runtime','app.evaluation.runner',
        'app.evaluation.recording','app.evaluation.scoring','app.evaluation.models')
    header.update(source_sha256={name:hashlib.sha256(Path(import_module(name).__file__).read_bytes()).hexdigest()
        for name in modules},benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        python_version=sys.version.split()[0],packages={name:version(name) for name in ('httpx','fastapi','pydantic')})
    (output_dir/'manifest.json').write_text(json.dumps(header,ensure_ascii=False,indent=2),encoding='utf-8')
    reports={}
    async def phase(session,name,*,count,reset=False):
        if reset and session.search_cache is None:raise RuntimeError('Cold phase requires its owned cache')
        manifest={**session.manifest,'cache_experiment_phase':name,
            'source_sha256':header['source_sha256'],'benchmark_sha256':header['benchmark_sha256'],
            'cache_reset_policy':'before_each_case' if reset else ('disabled' if name=='bypass' else 'shared_priming_and_warm_session')}
        return await evaluate_cases(client=session.client,workload=workload,recorder=session.recorder,
            capture=session.capture,paper_id=paper_id,output_dir=output_dir/name,manifest=manifest,repeats=count,
            before_case=session.search_cache.clear if reset else None)
    async with session_factory(settings=settings.model_copy(update={'web_search_cache_enabled':False}),
            workload=workload,paper_id=paper_id,database_url=database_url) as session:
        reports['bypass']=await phase(session,'bypass',count=repeats)
    enabled=settings.model_copy(update={'web_search_cache_enabled':True})
    async with session_factory(settings=enabled,workload=workload,paper_id=paper_id,database_url=database_url) as session:
        reports['cold']=await phase(session,'cold',count=repeats,reset=True)
    async with session_factory(settings=enabled,workload=workload,paper_id=paper_id,database_url=database_url) as session:
        reports['priming']=await phase(session,'priming',count=1)
        reports['warm']=await phase(session,'warm',count=repeats)
    observations={phase:[{'case_id':row['case_id'],'repeat':row['repeat'],'request_id':row['request_id'],
        'observation_complete':row['tool_observation_complete'],**_observed_web(row)} for row in report['results']]
        for phase,report in reports.items()}
    primed={r['case_id']:r for r in observations['priming']}
    baseline={(r['case_id'],r['repeat']):r for r in observations['bypass']}
    profile_match=all(bool(r['profiles']) and r['profiles']==primed[r['case_id']]['profiles'] for r in observations['warm'])
    source_match=all(r['source_snapshot_sha256'] is not None and
        r['source_snapshot_sha256']==primed[r['case_id']]['source_snapshot_sha256'] for r in observations['warm'])
    all_aligned=all(bool(r['profiles']) and r['profiles']==baseline[(r['case_id'],r['repeat'])]['profiles']
        for name in ('cold','warm') for r in observations[name])
    content_match=all(r['source_content_sha256'] is not None and
        r['source_content_sha256']==baseline[(r['case_id'],r['repeat'])]['source_content_sha256']
        for name in ('cold','priming','warm') for r in observations[name])
    complete=all(r['tool_observation_complete'] is True and r['automated_pass'] and r['metrics']['failed_tool_calls']==0
        for report in reports.values() for r in report['results'])
    fully_reused=all(r['metrics']['search_request_attempts']==0 and r['metrics'].get('search_cache_hits',0)==len(o['profiles'])
        and bool(o['profiles']) for r,o in zip(reports['warm']['results'],observations['warm']))
    summaries={name:_summary(report) for name,report in reports.items()}
    comparison={'all_warm_profiles_match_priming':profile_match,'all_warm_visible_sources_match_priming':source_match,
        'actual_profiles_match_bypass':all_aligned,'source_contents_match_bypass':content_match,'all_warm_searches_are_cache_hits':fully_reused,
        'paired_descriptive_comparison_eligible':profile_match and source_match and all_aligned and content_match and complete and fully_reused,
        'causal_or_stable_speedup_proven':False,
        'note':'Even aligned queries do not control model sampling, live provider changes or fixed phase order.'}
    hit_ages=[t['spans'][0]['attributes']['cache_age_seconds'] for r in reports['warm']['results'] for t in r['traces']
        if t['spans'] and t['spans'][0]['name']=='web.search' and t['spans'][0]['attributes'].get('cache_status')=='hit']
    result={**header,'finished_at':datetime.now(timezone.utc).isoformat(),'phases':summaries,'comparison':comparison,
        'observations':observations,'manual_review':{'status':'pending','source_support':'pending','relevance':'pending',
            'freshness':'pending','evidence_basis':'provider summaries visible to model, not fetched webpage bodies'},
        'limitations':['ASGI end-to-end within this process, not TCP deployment/load testing.',
            'Constrained prompts and small fixed-order sample do not represent unrestricted Agent traffic.',
            'Automatic checks and exact source reuse do not prove semantic answer correctness or current truth.',
            'LLM generation is not cached; priming cost and failed requests remain in their own phase.']}
    result['freshness_observations']={'cache_ttl_seconds':settings.web_search_cache_ttl_seconds,
        'warm_hit_ages_seconds':hit_ages,'all_observed_hits_younger_than_ttl':
            all(0<=age<settings.web_search_cache_ttl_seconds for age in hit_ages) if hit_ages else None,
        'current_source_truth_proven':False,'note':'TTL and cache ages do not prove Internet information is current.'}
    (output_dir/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    lines=['# Full Search Agent cache comparison','',
        '| Phase | API runs | Completed | HTTP failures | Checks pass | LLM turns | Search attempts | Hits | Body bytes | API median ms |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,p in summaries.items():
        lines.append(f"| {name} | {p['case_runs']} | {p['completed']} | {p['http_failures']} | {p['automatic_checks_passed']} | "
            f"{p['llm_turns']} | {p['search_attempts']} | {p['cache_hits']} | {p['response_body_bytes']} | {p['api_wall_ms_median']:.3f} |")
    lines+=['','Comparison checks:','']+['- '+key+': '+str(value) for key,value in comparison.items()]
    lines+=['','Semantic relevance/support/freshness review: pending.','']+['- '+v for v in result['limitations']]
    (output_dir/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result
