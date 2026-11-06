from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


ToolName=Literal['add_numbers','paper_retrieval','web_search']


class EvaluationCase(BaseModel):
    model_config=ConfigDict(strict=True,extra='forbid')
    id: str=Field(pattern=r'^[a-z0-9_-]{1,64}$')
    prompt: str=Field(min_length=1,max_length=4000)
    use_paper: bool=False
    enable_web_search: bool=False
    top_k: int=Field(default=5,ge=1,le=100)
    required_tools: list[ToolName]=Field(default_factory=list)
    allowed_tools: list[ToolName]=Field(default_factory=lambda:['add_numbers','paper_retrieval','web_search'])
    expected_pages: list[int]=Field(default_factory=list)
    expected_web_domains: list[str]=Field(default_factory=list)
    expected_insufficient: bool | None=None
    answer_terms: list[list[str]]=Field(default_factory=list)
    reference_notes: str=''

    @model_validator(mode='after')
    def consistent(self):
        if not self.prompt.strip():
            raise ValueError('Prompt must not be blank')
        if not set(self.required_tools).issubset(self.allowed_tools):
            raise ValueError('Required tools must be allowed')
        if 'paper_retrieval' in self.required_tools and not self.use_paper:
            raise ValueError('Paper tools require a selected paper')
        if 'web_search' in self.required_tools and not self.enable_web_search:
            raise ValueError('Search tools require web mode')
        if self.expected_pages and (not self.use_paper or any(type(p) is not int or p<=0 for p in self.expected_pages)):
            raise ValueError('Support pages require positive PDF sequence numbers and paper scope')
        if any(not group or any(not item.strip() for item in group) for group in self.answer_terms):
            raise ValueError('Term groups must not be empty')
        if any(not domain or '/' in domain or ':' in domain or domain.strip()!=domain for domain in self.expected_web_domains):
            raise ValueError('Expected source domains must be hostnames')
        return self

    def to_request(self,paper_id):
        if self.use_paper and not paper_id:
            raise ValueError('This workload requires an explicit paper_id')
        body={'prompt':self.prompt,'enable_web_search':self.enable_web_search,'top_k':self.top_k}
        if self.use_paper:
            body['paper_id']=paper_id
        return body


class Workload(BaseModel):
    model_config=ConfigDict(strict=True,extra='forbid')
    version: str=Field(min_length=1,max_length=100)
    paper_sha256: str=Field(pattern=r'^[a-f0-9]{64}$')
    cases: list[EvaluationCase]=Field(min_length=1,max_length=64)

    @model_validator(mode='after')
    def unique_ids(self):
        if len({case.id for case in self.cases})!=len(self.cases):
            raise ValueError('Evaluation case IDs must be unique')
        return self
