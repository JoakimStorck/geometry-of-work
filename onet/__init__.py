"""Read-only access layer for O*NET data under data/onet/.

Goals
- Notebooks should not hardcode file paths or file extensions.
- Version selection accepts '30_0' or '30.0' (list_versions returns dot form).
- Read-only: never writes to data/onet/.
"""

from .api import (
    list_versions,
    get_db,
    default_data_onet_dir,
    build_df_tasks,
    load_occ_meta,
    load_rle,
    load_job_families,
    load_crosswalk_2010_2019,
)

from .db import OnetDB

# Namespaces (batteries included)
from . import elements, tasks, occupations, skills, abilities, riasec

__all__ = [
    "list_versions",
    "get_db",
    "default_data_onet_dir",
    "OnetDB",
    "build_df_tasks",
    "load_occ_meta",
    "load_rle",
    "load_job_families",
    "load_crosswalk_2010_2019",
    "elements",
    "tasks",
    "occupations",
    "skills",
    "abilities",
    "riasec",
]
