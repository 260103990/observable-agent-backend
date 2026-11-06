"""Bounded English lexical queries and deterministic rank-only fusion."""
import math
import re

from app.rag.storage_models import RetrievedChunk


def lexical_query(query: str) -> str:
    if not isinstance(query,str) or not query.strip() or len(query)>4000:
        raise ValueError('Query must contain 1 to 4000 characters')
    # Only alphanumeric terms enter websearch syntax. Quoting/operators remain data.
    terms=list(dict.fromkeys(re.findall(r'[a-z0-9]+',query.lower())))[:64]
    return ' OR '.join(terms)


def fuse_rankings(vector_hits, lexical_hits, *, top_k: int):
    if type(top_k) is not int or not 1<=top_k<=100:
        raise ValueError('top_k must be an integer between 1 and 100')
    sources, scores, vector_positions = {}, {}, {}
    paper_id=None
    for list_index, hits in enumerate((vector_hits,lexical_hits)):
        seen=set()
        for hit in hits:
            if not isinstance(hit,RetrievedChunk) or not math.isfinite(hit.score):
                raise ValueError('Expected finite retrieved chunks')
            paper_id=paper_id or hit.paper_id
            if hit.paper_id!=paper_id:
                raise ValueError('Hybrid candidates must belong to one paper')
            key=(hit.paper_id,hit.chunk.page_number,hit.chunk.chunk_index)
            previous=sources.get(key)
            if previous is not None and previous!=hit:
                raise ValueError('Conflicting source or score for one chunk')
            sources.setdefault(key,hit)
            if key in seen:continue
            seen.add(key)
            rank=len(seen)
            scores[key]=scores.get(key,0.0)+1/(60+rank)
            if list_index==0:vector_positions[key]=rank
    ordered=sorted(sources,key=lambda key:(-scores[key],vector_positions.get(key,101),key))
    return [sources[key] for key in ordered[:top_k]]
