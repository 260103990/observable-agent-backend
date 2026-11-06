"""Explicit live TCP smoke test using only owned, database-free Uvicorn processes."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import httpx


CHILD = r'''
import os, sys, threading
import uvicorn
from app import main
from app.config import Settings, get_settings

# Never read .env or connect to a database, including when the caller normally does.
settings = Settings(_env_file=None).model_copy(update={
    'database_url':None, 'rag_enabled':False, 'web_search_enabled':False})
if sys.argv[2] == 'missing_key':
    settings = settings.model_copy(update={'deepseek_api_key':None})
main.get_settings = lambda: settings
main.app.dependency_overrides[get_settings] = lambda: settings
server = uvicorn.Server(uvicorn.Config(main.app,host='127.0.0.1',port=int(sys.argv[1]),log_level='warning'))
def stop_on_stdin():
    for line in sys.stdin:
        if line.strip() == 'stop':
            server.should_exit = True
            return
threading.Thread(target=stop_on_stdin,daemon=True).start()
server.run()
'''


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def unused_port():
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        return probe.getsockname()[1]


def closed(port):
    with socket.socket() as probe:
        probe.settimeout(0.5)
        return probe.connect_ex(('127.0.0.1', port)) != 0


def scenario(name, output, root):
    port = unused_port()
    record = {'scenario':name, 'database_enabled':False, 'port':port}
    process = None
    with (output / (name + '.log')).open('w', encoding='utf-8') as log:
        try:
            child_env = os.environ.copy()
            if name == 'missing_key':
                child_env.pop('DEEPSEEK_API_KEY', None)
            process = subprocess.Popen([sys.executable,'-B','-c',CHILD,str(port),name],
                cwd=root,env=child_env,stdin=subprocess.PIPE,stdout=log,stderr=log,
                text=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            with httpx.Client(base_url=f'http://127.0.0.1:{port}', trust_env=False, timeout=260) as http:
                deadline=time.monotonic()+15
                while True:
                    check(process.poll() is None, 'Owned server exited during startup')
                    try:
                        if http.get('/health', timeout=0.5).status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    check(time.monotonic()<deadline, 'Owned server startup deadline exceeded')
                    time.sleep(0.1)
                record['health_status']=200
                if name == 'live_calculator':
                    response=http.post('/agent/loop', json={'prompt':
                        'Use add_numbers exactly once to calculate 17 + 25, then report the result.'})
                    record['loop_http_status']=response.status_code
                    check(response.status_code==200, 'Live calculator did not return HTTP200')
                    body=response.json()
                    check(body['status']=='completed' and '42' in body['answer'], 'Unexpected calculator answer')
                    check(body['tool_calls']==1, 'Calculator did not execute exactly one tool')
                    check(response.headers.get('X-Trace-ID')==body['trace_id'], 'Trace header mismatch')
                    trace=http.get('/traces/'+body['trace_id'])
                    check(trace.status_code==200, 'Trace lookup failed')
                    spans=trace.json()['spans']
                    check(any(s['name']=='tool.execute' for s in spans), 'Missing tool span')
                    invalid=http.post('/agent/loop',json={'prompt':'','unrecognized':True})
                    check(invalid.status_code==422, 'Configured server did not reject invalid input')
                    record.update(status=body['status'],answer=body['answer'],rounds=body['rounds'],
                        tool_calls=body['tool_calls'],final_repairs=body['final_repairs'],
                        trace_lookup_http_status=trace.status_code,trace_spans=len(spans),
                        invalid_input_http_status=invalid.status_code)
                else:
                    response=http.post('/agent/loop',json={'prompt':'Calculate 17+25.'})
                    check(response.status_code==503, 'Missing key did not fail closed')
                    record['missing_key_http_status']=response.status_code
        finally:
            if process is not None:
                if process.poll() is None:
                    try:
                        process.stdin.write('stop\n')
                        process.stdin.flush()
                        process.wait(timeout=15)
                    except (BrokenPipeError,OSError,subprocess.TimeoutExpired):
                        # Only the exact process created above may be stopped.
                        process.terminate()
                        process.wait(timeout=10)
                        record['forced_termination']=True
                process.stdin.close()
                record['exit_code']=process.returncode
                record['port_closed']=closed(port)
                (output/(name+'.json')).write_text(json.dumps(record,ensure_ascii=True,indent=2),encoding='utf-8')
    check(record.get('exit_code')==0 and not record.get('forced_termination') and record['port_closed'],
          'Owned server did not shut down gracefully')
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    if not args.live:
        parser.error('--live is required for a real paid model request')
    if not os.getenv('DEEPSEEK_API_KEY','').strip():
        parser.error('Configure DEEPSEEK_API_KEY in the process environment')
    args.output.mkdir(parents=True, exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    records=[scenario(name,args.output,root) for name in ('live_calculator','missing_key')]
    (args.output/'report.json').write_text(json.dumps(records,ensure_ascii=True,indent=2),encoding='utf-8')
    print(json.dumps({'scenarios_passed':len(records),'graceful_shutdowns':len(records)}))


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        print('HTTP acceptance failed: '+type(exc).__name__+'. See local scenario reports.',file=sys.stderr)
        raise SystemExit(2) from None
