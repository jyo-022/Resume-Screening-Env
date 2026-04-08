from pydantic import BaseModel
from typing import List

class Action(BaseModel):
    ranked_candidates: List[str]
    flagged_candidates: List[str] = []