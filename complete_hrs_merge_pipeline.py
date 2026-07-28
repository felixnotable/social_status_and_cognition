from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# the dataset includes data from wave 6 to wave 13
WAVES = list(range(6, 14))
YEAR_BY_WAVE = {6: 2002, 7: 2004, 8: 2006, 9: 2008, 10: 2010, 11: 2012, 12: 2014, 13: 2016}

# map from the key metrics to the column names in h-hrs.dta
# section stress -> subsection loneliness -> a list of columns named in pattern of r{wave}variable_name
SOCIAL_MAP = {
    "loneliness3": "r{w}lnlys3",
    "lack_companionship": "r{w}complac",
    "left_out": "r{w}leftout",
    "feels_isolated": "r{w}isolate",
    "lives_alone": "h{w}lvalone",
    "weekly_contact_children": "r{w}kcnt",
    "weekly_contact_relatives_friends": "r{w}rfcnt",
    "weekly_social_activity": "r{w}socwk",
    "monthly_social_activity": "r{w}socmn",
    "orientation_date": "r{w}orient",
    "numeracy": "r{w}numer",
    "verbal_fluency": "r{w}verbf",
}

# gender option: integer to status map in RAND
SEX_MAP = {1: "Male", 2: "Female"}
# race option: integer to status map in RAND
RACE_MAP = {1: "White", 2: "Black/African American", 3: "Other"}
# marial status option: integer to status in RAND
MARITAL_MAP = {
    1: "Married",
    2: "Married, spouse absent",
    3: "Partnered",
    4: "Separated",
    5: "Divorced",
    6: "Separated/Divorced",
    7: "Widowed",
    8: "Never married",
}


def extract_member(zip_path: Path, member_ending: str, out_dir: Path) -> Path:
    """
    Extract the first member file in the zip whose name ends with member_ending.
    The member_ending can be a .dta, a .zip or else. 
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        matches = [n for n in zf.namelist() if n.lower().endswith(member_ending.lower())]
        if not matches:
            raise FileNotFoundError(f"No member ending {member_ending!r} in {zip_path}")
        member = matches[0]
        target = out_dir / Path(member).name
        if not target.exists():
            with zf.open(member) as src, target.open("wb") as dst:
                dst.write(src.read())
    return target


def prepare_sources(source_dir: Path, work_dir: Path) -> tuple[Path, Path, Path]:
    """Extract the three raw .dta files needed for the final merge."""
    # paths to the zip files
    predicted_outer = source_dir / "PredictedCognitionDementiaMeasures.zip"
    hhrs_zip = source_dir / "H_HRS_d_stata.zip"
    rand_zip = source_dir / "randhrs1992_2022v1_STATA.zip"

    # get the predicted coginition .dta
    nested = extract_member(predicted_outer, "Dementia_HRS_2000-2016_Basic_Release1_2m.zip", work_dir)
    predicted_dta = extract_member(nested, ".dta", work_dir)
    # get the HHRS .dta
    hhrs_dta = extract_member(hhrs_zip, "H_HRS_d.dta", work_dir)
    # get RAND .dta
    rand_dta = extract_member(rand_zip, "randhrs1992_2022v1.dta", work_dir)
    return predicted_dta, hhrs_dta, rand_dta


def read_predicted_cognition(predicted_dta: Path) -> pd.DataFrame:
    """
    read the predicted cognition .dta and extract the interested columns
    """
    cols = ["hhidpn", "wave", "Cog", "CogSd", "PrDem", "PrCIND", "PrNorm"]
    cognition = pd.read_stata(predicted_dta, columns=cols, convert_categoricals=False)
    cognition = cognition[cognition["wave"].isin(WAVES)].copy()
    # convert columns to int64 type
    cognition["hhidpn"] = cognition["hhidpn"].astype("int64")
    cognition["wave"] = cognition["wave"].astype("int64")
    # deduplicate if multiple rows exist for the same "hhidpn"+"wave"
    cognition = cognition.drop_duplicates(["hhidpn", "wave"], keep="first")
    return cognition


def read_harmonized_social(hhrs_dta: Path) -> pd.DataFrame:
    """
    Read the harmonized social file, extract the interested columns matching naming template 
    defined in SOCIAL_MAP.values()
    """
    from pandas.io.stata import StataReader
    # get all of the availabe columns in the HHRS dta
    available = set(StataReader(hhrs_dta, convert_categoricals=False).variable_labels())
    # requires is supposed to contain a list of interested columns. 
    # "hhidpn" column is always required.
    required = ["hhidpn"]
    for wave in WAVES:
        # wave 6-13
        for template in  SOCIAL_MAP.values():
            # replace r{w}xxxx with wave, e.g. r{w}compac -> r6compac
            candidate = template.format(w=wave)
            if candidate in available:
                # if the interested column exists, add it to the required column
                required.append(candidate)

    # read out all the interested columns
    wide = pd.read_stata(hhrs_dta, columns=required, convert_categoricals=False)
    # convert "hhidpn" into int64 
    wide["hhidpn"] = wide["hhidpn"].astype("int64")

    # re-format the data to hhidpn+wave records
    # original: hhidpn, r{w}xxx, ....
    # after re-format: 
    #    hhidpn, 6, r6xxx, 
    #    hhidpn, 7, r7xxx, 
    frames: list[pd.DataFrame] = []
    for wave in WAVES:
        rename = {
            template.format(w=wave): final
            # "r6lyln3":"loneliness3"
            # "r7lyln3": "longliness3"
            # "r13lyln3": "longliness3"
            # "r6complac": "lack_companionship"
            # "r13complac": "lack_companionship"
            # ....
            ## map of <dta column name> to final csv column name
            for final, template in SOCIAL_MAP.items()
            if template.format(w=wave) in wide.columns
        }

        piece = wide[["hhidpn", *rename.keys()]].rename(columns=rename).copy()
        for final in SOCIAL_MAP:
            if final not in piece.columns:
                piece[final] = np.nan
        piece = piece[["hhidpn", *SOCIAL_MAP.keys()]]
        piece["wave"] = wave
        frames.append(piece)

    social = pd.concat(frames, ignore_index=True)
    social = social.drop_duplicates(["hhidpn", "wave"], keep="first")
    return social


def read_rand_covariates(rand_dta: Path) -> pd.DataFrame:
    # create interested columns. 
    # first, the columns that do not change with wave
    selected = ["hhidpn", "ragender", "raedyrs", "raracem", "rahispan"]
    # then, the columns that change with wave
    for wave in WAVES:
        selected.extend([
            f"r{wave}agey_e",
            f"r{wave}iwendy",
            f"h{wave}atotw",
            f"r{wave}mstat",
        ])
    # get raw data with the interested columns
    rand = pd.read_stata(rand_dta, columns=selected, convert_categoricals=False)
    rand["hhidpn"] = rand["hhidpn"].astype("int64")

    # re-format the data to hhidpn+wave records
    # original: hhidpn, r{w}xxx, ragender....
    # after re-format: 
    #    hhidpn, 6, r6xxx, ragender
    #    hhidpn, 7, r7xxx, ragender
    frames: list[pd.DataFrame] = []
    for wave in WAVES:
        piece = rand[[
            "hhidpn", "ragender", "raedyrs", "raracem", "rahispan",
            f"r{wave}agey_e", f"r{wave}iwendy", f"h{wave}atotw", f"r{wave}mstat",
        ]].copy()
        piece.columns = [
            "hhidpn", "ragender", "raedyrs", "raracem", "rahispan",
            "age", "year", "wealth", "mstat_code",
        ]
        piece["wave"] = wave
        frames.append(piece)

    long = pd.concat(frames, ignore_index=True)
    long = long.drop_duplicates(["hhidpn", "wave"], keep="first")
    return long


def label_covariates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Re-label the RAND table, and convert the integer value to words 
    """
    out = df.copy()
    out["sex"] = out["ragender"].map(SEX_MAP)
    out["education"] = out["raedyrs"]
    out["race"] = out["raracem"].map(RACE_MAP)
    out["hispanic"] = np.select(
        [out["rahispan"].eq(1), out["rahispan"].eq(0)],
        ["Hispanic", "Not Hispanic"],
        default=None,
    )
    out["race_ethnicity"] = np.where(
        out["rahispan"].eq(1),
        "Hispanic",
        np.where(
            out["rahispan"].eq(0) & out["race"].notna(),
            out["race"] + ", non-Hispanic",
            None,
        ),
    )
    out["marital_status"] = out["mstat_code"].map(MARITAL_MAP)
    return out


def validate(df: pd.DataFrame) -> dict:
    """
    Collect the necessary statistics information. 
    """
    key_dupes = int(df.duplicated(["hhidpn", "wave"]).sum())
    probability_sum = df[["PrDem", "PrCIND", "PrNorm"]].sum(axis=1, min_count=3)
    probability_deviation = float((probability_sum - 1).abs().max())

    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "unique_respondents": int(df["hhidpn"].nunique()),
        "waves": sorted(int(x) for x in df["wave"].dropna().unique()),
        "duplicate_primary_keys": key_dupes,
        "missing_by_column": {c: int(df[c].isna().sum()) for c in df.columns},
        "nonmissing_by_column": {c: int(df[c].notna().sum()) for c in df.columns},
        "max_abs_probability_sum_deviation": probability_deviation,
    }


def build_dataset(predicted_dta: Path, hhrs_dta: Path, rand_dta: Path) -> pd.DataFrame:
    """
    Read interested columns from multiple dataset, and convert raw data to unified hhidpn+wave format.
    Merge the 3 tables with hhidpn+wave as primary key. 
    """
    cognition = read_predicted_cognition(predicted_dta)
    social = read_harmonized_social(hhrs_dta)
    rand = label_covariates(read_rand_covariates(rand_dta))

    merged = cognition.merge(
        social,
        on=["hhidpn", "wave"],
        how="left",
        validate="one_to_one",
    )
    merged = merged.merge(
        rand,
        on=["hhidpn", "wave"],
        how="left",
        validate="one_to_one",
    )

    # The original project used interview year from RAND. For any rare missing
    # year, retain the deterministic HRS wave-year mapping as a fallback.
    merged["year"] = merged["year"].fillna(merged["wave"].map(YEAR_BY_WAVE))

    final_columns = [
        "hhidpn", "wave", "year", "age",
        "Cog", "CogSd", "PrDem", "PrCIND", "PrNorm",
        "loneliness3", "lack_companionship", "left_out", "feels_isolated",
        "lives_alone", "weekly_contact_children",
        "weekly_contact_relatives_friends", "weekly_social_activity",
        "monthly_social_activity", "orientation_date", "numeracy",
        "verbal_fluency", "sex", "education", "wealth", "race",
        "hispanic", "race_ethnicity", "marital_status",
    ]
    return merged[final_columns].sort_values(["hhidpn", "wave"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild the HRS cognition-loneliness analysis dataset.")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=Path("work"))
    parser.add_argument("--output", type=Path, default=Path("merged_cognition_loneliness_HRS_w6_w13_with_demographics_wealth.csv"))
    parser.add_argument("--audit", type=Path, default=Path("merged_dataset_validation.json"))
    args = parser.parse_args()

    predicted_dta, hhrs_dta, rand_dta = prepare_sources(args.source_dir, args.work_dir)
    final = build_dataset(predicted_dta, hhrs_dta, rand_dta)
    final.to_csv(args.output, index=False, encoding="utf-8-sig")

    audit = validate(final)
    args.audit.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"Wrote {args.output} ({len(final):,} rows x {final.shape[1]} columns)")
    print(f"Wrote {args.audit}")


if __name__ == "__main__":
    main()
