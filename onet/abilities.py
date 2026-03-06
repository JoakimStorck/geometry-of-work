from __future__ import annotations

import pandas as pd

from .db import OnetDB
from . import elements

DEFAULT_SCALE_PREFERENCE = ("LV", "IM")

def raw(db: OnetDB) -> pd.DataFrame:
    return db.read("Abilities")

def wide(db: OnetDB, *, scale_preference=DEFAULT_SCALE_PREFERENCE) -> tuple[pd.DataFrame, str]:
    return elements.wide(db, "Abilities", scale_preference=scale_preference)

def long(db: OnetDB, *, scale_preference=DEFAULT_SCALE_PREFERENCE) -> tuple[pd.DataFrame, str]:
    return elements.long(db, "Abilities", scale_preference=scale_preference, element_name="ability", value_name="value")
