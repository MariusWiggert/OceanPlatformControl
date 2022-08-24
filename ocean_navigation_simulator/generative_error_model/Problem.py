from dataclasses import dataclass
from typing import List
import datetime

@dataclass
class Problem:
    lon_range: List[float]
    lat_range: List[float]
    t_range: List[datetime.datetime]
