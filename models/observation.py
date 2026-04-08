from pydantic import BaseModel
from typing import List, Dict

class Observation(BaseModel):
    job_description: Dict
    candidates: List[Dict]
    step: int
    history: List[Dict] = []