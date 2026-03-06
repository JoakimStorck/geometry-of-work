from __future__ import annotations

import pandas as pd

from .db import OnetDB
from .transforms import load_occ_meta as meta

def data(db: OnetDB) -> pd.DataFrame:
    return db.read("Occupation Data")

def occ_meta(db: OnetDB) -> pd.DataFrame:
    """Standard occupation meta: onet_code, Title, Job Family."""
    return meta(db)
