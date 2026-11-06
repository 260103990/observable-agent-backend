"""Normalize one complete final-answer wrapper without changing strict tool JSON."""
import re

from app.agent.loop_models import load_json_object


_FINAL_FENCE=re.compile(r'```(?:json)?[ \t]*\r?\n(?P<body>[\s\S]*?)\r?\n```',re.IGNORECASE)


def load_final_json_object(content: str) -> dict:
    if not isinstance(content,str):
        raise ValueError('Final answer must be text')
    value=content.strip()
    if value.startswith('```'):
        match=_FINAL_FENCE.fullmatch(value)
        if match is None:
            raise ValueError('Expected one complete JSON code fence')
        value=match.group('body')
    return load_json_object(value)
