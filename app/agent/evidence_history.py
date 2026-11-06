"""Lossless, request-local model history projection for owned evidence tools."""
from copy import deepcopy
from dataclasses import dataclass
import json
import re

from app.agent.loop_models import load_json_object


EVIDENCE_HISTORY_PROMPT = '''
In tool history an evidence entry containing only evidence_id and same_as_tool_call_id
is an exact duplicate of the full entry with that ID in the indicated later tool message.
Read that full entry as its source; this does not create new evidence or change citation IDs.
'''
OWNED_EVIDENCE_TOOLS = {'paper_retrieval': 'E', 'web_search': 'W'}


def _encode(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


@dataclass(frozen=True)
class HistoryProjection:
    messages: list
    raw_chars: int
    chars: int
    removed_evidence_copies: int


def _evidence_messages(messages, evidence_tools):
    """Only inspect successful, well-formed owned-tool envelopes; leave all else alone."""
    calls = {}
    for message in messages:
        if message.get('role') == 'assistant':
            for call in message.get('tool_calls') or []:
                calls[call['id']] = call['function']['name']
    for index, message in enumerate(messages):
        if message.get('role') != 'tool':
            continue
        call_id = message.get('tool_call_id')
        name = calls.get(call_id)
        if name not in evidence_tools or name not in OWNED_EVIDENCE_TOOLS:
            continue
        try:
            result = load_json_object(message['content'])
        except (ValueError, TypeError, RecursionError):
            continue
        output = result.get('output')
        if result.get('ok') is not True or not isinstance(output, dict):
            continue
        items = output.get('evidence')
        if not isinstance(items, list):
            continue
        identifiers = [item.get('evidence_id') if isinstance(item, dict) else None for item in items]
        pattern = OWNED_EVIDENCE_TOOLS[name] + r'[1-9][0-9]*'
        if (any(not isinstance(identifier, str) or not re.fullmatch(pattern, identifier) for identifier in identifiers)
                or len(set(identifiers)) != len(identifiers)):
            continue
        yield index, call_id, name, result


def project_history(messages, *, evidence_tools=()):
    """Keep the latest full copy, replacing only byte-equivalent JSON evidence items.

    Original messages are never changed. No summarization, deletion of turns or
    merging by URL, meaning or evidence ID alone is permitted.
    """
    view = deepcopy(messages)
    raw_chars = len(_encode(messages))
    latest = {}
    removed = 0
    for index, call_id, name, result in reversed(list(_evidence_messages(view, evidence_tools))):
        items = result['output']['evidence']
        replacements = 0
        for position, item in enumerate(items):
            if 'same_as_tool_call_id' in item:
                continue
            key = (name, item['evidence_id'], json.dumps(item, sort_keys=True, ensure_ascii=False, allow_nan=False))
            target = latest.get(key)
            reference = {'evidence_id': item['evidence_id'], 'same_as_tool_call_id': target}
            if target is not None and len(_encode(reference)) < len(_encode(item)):
                items[position] = reference
                replacements += 1
            else:
                latest[key] = call_id
        if replacements:
            content = _encode(result)
            # Preserve original formatting if rewriting the envelope would outweigh savings.
            if len(content) < len(view[index]['content']):
                view[index]['content'] = content
                removed += replacements
    return HistoryProjection(view, raw_chars, len(_encode(view)), removed)


def restore_evidence_history(messages):
    """Evaluation-only reconstruction of complete results from a single projection.

    References must point directly to a later full entry of the same owned tool.
    Reject broken references rather than recording an incomplete observation.
    """
    restored = deepcopy(messages)
    records = list(_evidence_messages(restored, OWNED_EVIDENCE_TOOLS))
    lookup = {(call_id, item['evidence_id']): (index, name, item)
              for index, call_id, name, result in records for item in result['output']['evidence']
              if 'same_as_tool_call_id' not in item}
    for index, call_id, name, result in records:
        changed = False
        for position, item in enumerate(result['output']['evidence']):
            if 'same_as_tool_call_id' not in item:
                continue
            if set(item) != {'evidence_id', 'same_as_tool_call_id'} or not isinstance(item['same_as_tool_call_id'], str):
                raise ValueError('Invalid evidence history reference')
            target = lookup.get((item['same_as_tool_call_id'], item['evidence_id']))
            if target is None or target[0] <= index or target[1] != name:
                raise ValueError('Evidence history reference has no later same-tool source')
            result['output']['evidence'][position] = deepcopy(target[2])
            changed = True
        if changed:
            restored[index]['content'] = _encode(result)
    return restored
