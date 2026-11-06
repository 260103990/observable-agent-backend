from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from app.agent.loop_models import LoopResult
from app.evaluation.scoring import score_case, summarize_traces


async def evaluate_cases(*,client,workload,recorder,capture,paper_id,output_dir,manifest,repeats=1,before_case=None):
    if type(repeats) is not int or not 1<=repeats<=5:
        raise ValueError('Repeat count must be between 1 and 5')
    bodies={case.id:case.to_request(paper_id) for case in workload.cases}
    output_dir=Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    names=('manifest.json','cases.jsonl','report.json','report.md')
    if any((output_dir/name).exists() for name in names):
        raise FileExistsError('Choose a fresh evaluation output directory')
    run_id=str(uuid4())
    started=datetime.now(timezone.utc).isoformat()
    header={'run_id':run_id,'started_at':started,'workload_version':workload.version,
            'workload':workload.model_dump(),'manifest':manifest,'repeats':repeats}
    with (output_dir/'manifest.json').open('x',encoding='utf-8') as file:
        json.dump(header,file,ensure_ascii=False,indent=2)
    rows=[]
    with (output_dir/'cases.jsonl').open('x',encoding='utf-8') as journal:
        for repeat in range(1,repeats+1):
            for case in workload.cases:
                if before_case is not None:
                    before_case()
                request_id=f'{run_id}-{case.id}-{repeat}'
                started_at=perf_counter()
                result,error_type,http_status,root_trace_id={},None,None,None
                try:
                    with capture.for_request(request_id):
                        response=await client.post('/agent/loop',json=bodies[case.id],headers={'X-Request-ID':request_id})
                    http_status=response.status_code
                    root_trace_id=response.headers.get('X-Trace-ID')
                    if http_status==200:
                        payload=response.json()
                        result={**LoopResult.model_validate({k:v for k,v in payload.items() if k!='trace_id'}).model_dump(),
                                'trace_id':payload['trace_id']}
                except Exception as exc:
                    error_type=type(exc).__name__
                wall_ms=(perf_counter()-started_at)*1000
                traces=await recorder.for_request(request_id)
                observed=capture.for_case(request_id)
                candidates=[t.trace_id for t in traces if t.spans and t.spans[0].name=='agent.loop']
                if root_trace_id is None and len(candidates)==1:
                    root_trace_id=candidates[0]
                metrics=summarize_traces(traces,root_trace_id=root_trace_id or result.get('trace_id'))
                checked=score_case(case,result,metrics,observed)
                checked['checks']['http_success']=http_status==200 and error_type is None
                checked['automated_pass']=checked['automated_pass'] and checked['checks']['http_success']
                row={'case_id':case.id,'repeat':repeat,'request_id':request_id,'request':bodies[case.id],
                     'http_status':http_status,'trace_id':root_trace_id,'error_type':error_type,'api_wall_ms':wall_ms,
                     'result':result,'metrics':metrics,'observations':observed,
                     'traces':[t.model_dump() for t in traces],**checked}
                rows.append(row)
                journal.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
                journal.flush()
    report={**header,'finished_at':datetime.now(timezone.utc).isoformat(),'results':rows,
            'case_runs':len(rows),'automated_pass_count':sum(row['automated_pass'] for row in rows),
            'manual_review_pending':len(rows),
            'limitations':['ASGI in-process API evaluation, not a uvicorn/TCP deployment test.',
                'Provider labels and live search content can change between runs.',
                'Model-visible support pages are after evidence budgets, not complete index Top-K recall.',
                'Search body bytes exclude headers/TLS/request bytes and do not equal wire traffic.']}
    with (output_dir/'report.json').open('x',encoding='utf-8') as file:
        json.dump(report,file,ensure_ascii=False,indent=2,allow_nan=False)
    lines=['# V2 fixed-workload evaluation','',f'Run: {run_id}',
           f'Automated checks: {report["automated_pass_count"]}/{len(rows)}; semantic review pending: {len(rows)}.',
           '', '| Case | Repeat | HTTP | Checks | Rounds | Tool calls | Search attempts | Body bytes | API ms |',
           '|---|---:|---:|---|---:|---:|---:|---:|---:|']
    for row in rows:
        data,metrics=row['result'],row['metrics']
        lines.append(f'| {row["case_id"]} | {row["repeat"]} | {row["http_status"]} | '
            f'{"pass" if row["automated_pass"] else "review"} | {data.get("rounds", "-")} | '
            f'{data.get("tool_calls", "-")} | {metrics["search_request_attempts"]} | '
            f'{metrics["search_response_body_bytes"]} | {row["api_wall_ms"]:.1f} |')
    for row in rows:
        failures=[key for key,value in row['checks'].items() if value is False]
        lines.extend(['',f'## {row["case_id"]} / {row["repeat"]}','',
            'Failed automatic checks: '+(', '.join(failures) or 'none'),
            '','Answer (model text, semantic support still requires review):',''])
        lines.extend('    '+line for line in (row['result'].get('answer') or '(no answer)').splitlines())
    lines.extend(['','## Limits','',*('- '+item for item in report['limitations'])])
    with (output_dir/'report.md').open('x',encoding='utf-8') as file:
        file.write('\n'.join(lines)+'\n')
    return report
