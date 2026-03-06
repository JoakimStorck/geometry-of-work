from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .db import OnetDB
from .io import read_csv

def load_job_families(db: OnetDB) -> pd.DataFrame:
    fp = Path(db.data_onet_dir) / "All_Job_Families.csv"
    if not fp.exists():
        raise FileNotFoundError(f"Missing: {fp}")
    df = read_csv(fp)
    # normalize
    if "Code" in df.columns:
        df["Code"] = df["Code"].astype(str).str.strip()
    return df

def load_crosswalk_2010_2019(db: OnetDB) -> Dict[str, str]:
    fp = Path(db.data_onet_dir) / "2010_to_2019_Crosswalk.csv"
    if not fp.exists():
        return {}
    cw = read_csv(fp)
    a = "O*NET-SOC 2010 Code"
    b = "O*NET-SOC 2019 Code"
    if a not in cw.columns or b not in cw.columns:
        return {}
    return cw.set_index(a)[b].astype(str).to_dict()

def _job_family_lookup(db: OnetDB) -> Dict[str, str]:
    job_fam = load_job_families(db)
    if "Code" not in job_fam.columns or "Job Family" not in job_fam.columns:
        return {}
    return dict(zip(job_fam["Code"].astype(str), job_fam["Job Family"].astype(str)))

def load_occ_meta(db: OnetDB) -> pd.DataFrame:
    occ = db.read("Occupation Data")[["O*NET-SOC Code", "Title"]].copy()
    occ["O*NET-SOC Code"] = occ["O*NET-SOC Code"].astype(str).str.strip()

    fam_lu = _job_family_lookup(db)
    cw_map = load_crosswalk_2010_2019(db)

    def map_family(code: str) -> str:
        code = str(code).strip()
        fam = fam_lu.get(code)
        if not fam and cw_map:
            fam = fam_lu.get(str(cw_map.get(code, "")).strip())
        return fam or "NA"

    out = (
        occ.rename(columns={"O*NET-SOC Code": "onet_code"})
           .drop_duplicates("onet_code")[["onet_code", "Title"]]
           .reset_index(drop=True)
    )
    out["Job Family"] = out["onet_code"].map(map_family)
    return out

def load_rle(db: OnetDB) -> pd.DataFrame:
    edu = db.read("Education, Training, and Experience")[
        ["O*NET-SOC Code", "Element Name", "Scale ID", "Category", "Data Value"]
    ].copy()

    df_rle = edu.loc[
        (edu["Element Name"] == "Required Level of Education") &
        (edu["Scale ID"] == "RL")
    ].copy()

    df_rle["Level"] = pd.to_numeric(df_rle["Category"], errors="coerce").astype("Int64")
    df_rle["Data Value"] = pd.to_numeric(df_rle["Data Value"], errors="coerce")
    df_rle.dropna(subset=["Level", "Data Value"], inplace=True)
    df_rle["Weighted"] = df_rle["Level"].astype(float) * df_rle["Data Value"]

    grp = (
        df_rle.groupby("O*NET-SOC Code", as_index=False)
              .agg(rle_sum=("Weighted", "sum"), pct_sum=("Data Value", "sum"))
    )
    grp["rle_mean"] = grp["rle_sum"] / grp["pct_sum"]
    out = grp.rename(columns={"O*NET-SOC Code": "onet_code"})[["onet_code", "rle_mean"]]
    out["onet_code"] = out["onet_code"].astype(str).str.strip()
    return out

def build_df_tasks(
    db: OnetDB,
    *,
    include_supplemental: bool = True,
    require_rle: bool = True,
    rt_scale_id: str = "RT",
) -> pd.DataFrame:
    """Build a task-level dataframe used for embeddings/PCA."""
    occ_meta = load_occ_meta(db)

    tasks = db.read("Task Statements")[["O*NET-SOC Code", "Task ID", "Task", "Task Type"]].copy()
    tasks["O*NET-SOC Code"] = tasks["O*NET-SOC Code"].astype(str).str.strip()
    tasks["Task"] = tasks["Task"].astype(str).str.strip()
    tasks["Task ID"] = pd.to_numeric(tasks["Task ID"], errors="raise").astype(int)
    tasks["Task Type"] = tasks["Task Type"].astype(str).str.strip().str.title()
    tasks["is_core"] = tasks["Task Type"].eq("Core")

    if not include_supplemental:
        tasks = tasks.loc[tasks["is_core"]].copy()

    from . import tasks as tasks_api
    ratings = tasks_api.ratings(db, normalize=False)[["O*NET-SOC Code", "Task ID", "Scale ID", "Data Value"]].copy()

    ratings_rt = (
        ratings.loc[ratings["Scale ID"] == rt_scale_id, ["O*NET-SOC Code", "Task ID", "Data Value"]]
              .rename(columns={"Data Value": "rt"})
    )
    ratings_rt["Task ID"] = pd.to_numeric(ratings_rt["Task ID"], errors="raise").astype(int)

    df = (
        tasks[["O*NET-SOC Code", "Task ID", "Task", "Task Type", "is_core"]]
        .merge(ratings_rt, on=["O*NET-SOC Code", "Task ID"], how="inner")
        .rename(columns={"O*NET-SOC Code": "onet_code"})
    )

    rle = load_rle(db)
    df = df.merge(rle, on="onet_code", how="left").merge(occ_meta, on="onet_code", how="left")

    if require_rle:
        df = df.dropna(subset=["rle_mean"]).copy()

    return df.reset_index(drop=True)
