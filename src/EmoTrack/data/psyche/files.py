"""Locate user-supplied sources for optional trajectory generation."""

from pathlib import Path

PSYCHE_SOURCE = "https://zenodo.org/records/5085146"
NHANES_SOURCE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DPQ_L.htm"


def require_psyche_files(raw_dir: Path) -> Path:
    path = raw_dir / "data_quarterly.parquet"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Obtain anon_processed_df_parquet from {PSYCHE_SOURCE} "
            "under its terms and save it here as data_quarterly.parquet."
        )
    return path


def require_nhanes_dpq_xpt(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Obtain DPQ_L.xpt from {NHANES_SOURCE} and place it here."
        )
    return path
