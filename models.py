from typing import List, Optional
from pydantic import BaseModel

class InferenceInput(BaseModel):
    data: List[float]  # Adjust this type to match the structure of input data
    pocket: List[float]
    idx: List[int]
    save_attention: bool = False
    val_rate: float = 1.0