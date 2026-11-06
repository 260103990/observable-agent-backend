from collections import Counter
import re
from urllib.parse import urlsplit


def summarize_traces(traces,*,root_trace_id):
    root=next((t for t in traces if t.trace_id==root_trace_id and t.spans and t.spans[0].name=='agent.loop'),None)
    spans=root.spans if root else []
    tool_spans=[s for s in spans if s.name=='tool.execute']
    searches=[t for t in traces if t.spans and t.spans[0].name=='web.search']
    return {
        'root_trace_present':root is not None,
        'logical_tool_calls':root.spans[0].attributes.get('tool_calls',len(tool_spans)) if root else None,
        'root_duration_ms':root.duration_ms if root else None,
        'model_duration_ms':sum(s.duration_ms for s in spans if s.name=='model.turn'),
        'model_turns':sum(s.name=='model.turn' for s in spans),
        'tool_duration_ms':sum(s.duration_ms for s in tool_spans),
        'tool_calls_by_name':dict(Counter(s.attributes.get('tool_name','unregistered') for s in tool_spans)),
        'failed_tool_calls':sum(s.status!='ok' for s in tool_spans),
        'search_trace_count':len(searches),
        'search_coalesced_calls':sum(t.spans[0].attributes.get('coalescing_status')=='reused' for t in searches),
        'search_coalescing_waits':sum(t.spans[0].attributes.get('coalescing_role')=='follower' for t in searches),
        'search_coalescing_capacity_bypasses':sum(t.spans[0].attributes.get('coalescing_status')=='capacity_bypass' for t in searches),
        'search_coalescing_wait_ms':sum(s.duration_ms for t in searches for s in t.spans if s.name=='search.coalescing.wait'),
        'search_cache_hits':sum(t.spans[0].attributes.get('cache_status')=='hit' for t in searches),
        'search_cache_misses':sum(t.spans[0].attributes.get('cache_status')=='miss' for t in searches),
        'search_cache_bypasses':sum(t.spans[0].attributes.get('cache_status','bypass')=='bypass' for t in searches),
        'search_cache_errors':sum('cache_error_type' in t.spans[0].attributes for t in searches),
        'search_request_attempts':sum(t.spans[0].attributes.get('outbound_request_attempts',0) for t in searches),
        'search_response_body_bytes':sum(t.spans[0].attributes.get('response_body_bytes',0) for t in searches),
        'search_http_duration_ms':sum(s.duration_ms for t in searches for s in t.spans if s.name=='search.http'),
        'paper_retrieval_trace_count':sum(bool(t.spans) and t.spans[0].name=='rag.retrieve' for t in traces),
    }


def score_case(case,result,metrics,observations):
    tools=set(metrics.get('tool_calls_by_name',{}))
    successful_tools={row.get('name') for row in observations.get('tool_results',[]) if row.get('result',{}).get('ok') is True}
    visible_pages={e['page_number'] for row in observations.get('tool_results',[])
        if row.get('name')=='paper_retrieval' and row.get('result',{}).get('ok')
        for e in row['result']['output'].get('evidence',[]) if 'page_number' in e}
    cited_pages={item['page_number'] for item in result.get('citations',[]) if 'page_number' in item}
    expected_pages=set(case.expected_pages)
    domains={urlsplit(item.get('url','')).hostname for item in result.get('web_citations',[])}
    def has_term(term):
        answer=result.get('answer') or ''
        if term.isdigit():
            return re.search(r'(?<!\d)'+re.escape(term)+r'(?!\d)',answer) is not None
        return term.casefold() in answer.casefold()
    checks={
        'completed':result.get('status')=='completed',
        'root_trace_present':metrics.get('root_trace_present',False),
        'required_tools_present':set(case.required_tools).issubset(tools),
        'required_tools_succeeded':set(case.required_tools).issubset(successful_tools),
        'only_allowed_tools_used':tools.issubset(case.allowed_tools),
        'model_visible_support_pages_present':expected_pages.issubset(visible_pages) if expected_pages else None,
        'cited_support_pages_present':expected_pages.issubset(cited_pages) if expected_pages else None,
        'expected_web_domain_present':any(host and (host==domain or host.endswith('.'+domain))
            for host in domains for domain in case.expected_web_domains) if case.expected_web_domains else None,
        'insufficiency_matches':result.get('insufficient_evidence')==case.expected_insufficient
            if case.expected_insufficient is not None else None,
        'answer_terms_present':all(any(has_term(term) for term in group) for group in case.answer_terms)
            if case.answer_terms else None,
    }
    expected_calls=metrics.get('logical_tool_calls')
    if expected_calls is None:
        expected_calls=result.get('tool_calls')
    observed=observations.get('tool_results',[])
    identifiers=[row.get('call_id') for row in observed]
    complete=(len(observed)==expected_calls and len(set(identifiers))==len(identifiers)
              and all(identifier is not None for identifier in identifiers)
              and not observations.get('capture_errors')) if type(expected_calls) is int else None
    checks['tool_observation_complete']=complete
    return {'checks':checks,'automated_pass':all(v for v in checks.values() if v is not None),
        'model_visible_pdf_pages':sorted(visible_pages),'cited_pdf_pages':sorted(cited_pages),
        'tool_observation_complete':complete,
        'manual_review':{'status':'pending','source_support':'pending','completeness':'pending',
                         'note':'Automated term/page/tool checks do not prove semantic correctness.'}}
