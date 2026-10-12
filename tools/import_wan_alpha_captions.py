#!/usr/bin/env python3
"""Import REAL Wan-Alpha T2V captions by exact foreground video basename.

The Wan-Alpha official training CSV already has 'prompt' strings! No need to
invent captions or run Qwen image captioning. Match case-insensitive MP4
stem, not weak fuzzy/semantic guesses; sample IDs resemble original stems.
Multiple captions for same clip: choose first in CSV, keep duplicate count.
Fail closed when required training/selection records have no exact caption.
"""
import argparse
import csv
import io
import json
from pathlib import Path

# Allow execution by absolute filename from /content Colab, not just repo cwd.
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from urllib.request import urlopen


WAN_T2V_CSV = (
    "https://raw.githubusercontent.com/WeChatCV/Wan-Alpha/"
    "main/Wan-Alpha-train/data.csv"
)


def normalize_key(value):
    return Path(str(value).replace("\\","/")).stem.casefold()


def match_captions(records,csv_text):
    rows=csv.DictReader(io.StringIO(csv_text))
    if not {"video_fgr","video_pha","prompt"}.issubset(rows.fieldnames or ()):
        raise ValueError("Official WanAlpha CSV lacks paired video and prompt columns")
    lookups={}
    duplicates=0
    for row in rows:
        prompt=(row.get("prompt") or "").strip()
        if not prompt:
            continue
        key=normalize_key(row["video_fgr"])
        if key in lookups:
            duplicates+=1
            continue
        lookups[key]=prompt
    mapped={}
    missing=[]
    for record in records:
        key=normalize_key(record.get("foreground",record.get("id","")))
        prompt=lookups.get(key) or lookups.get(normalize_key(record["id"]))
        if prompt:
            mapped[record["id"]]=prompt
        else:
            missing.append(record["id"])
    return mapped,missing,duplicates


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--source-csv",default=None,
                   help="Use a locally cached official CSV, otherwise download it")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--allow-missing",action="store_true",
                   help="Report missing required captions rather than failing")
    args=p.parse_args()
    records=json.loads(Path(args.manifest).read_text())["clips"]
    if args.source_csv:
        csv_text=Path(args.source_csv).read_text(encoding="utf-8-sig")
    else:
        print("DOWNLOADING_OFFICIAL_WAN_ALPHA_T2V_CAPTIONS",WAN_T2V_CSV,flush=True)
        with urlopen(WAN_T2V_CSV,timeout=45) as res:
            csv_text=res.read().decode("utf-8-sig")
    mapped,missing,duplicates=match_captions(records,csv_text)
    from run_vae_22gb import training_split
    train,val=training_split(records,args.seed)
    required={r["id"] for r in train+val[:3]}
    missing_required=sorted(required-set(mapped))
    out=Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(mapped,ensure_ascii=False,indent=2))
    print("OFFICIAL_WAN_CAPTION_COVERAGE",json.dumps({
        "matched":len(mapped),"total":len(records),
        "matched_required":len(required)-len(missing_required),
        "required":len(required),
        "official_multiple_caption_rows":duplicates,
        "missing_required":missing_required[:20],
        "saved_to":str(out),
        "note":"Only exact basename matching; no invented/fuzzy captions",
    },ensure_ascii=False),flush=True)
    if missing_required and not args.allow_missing:
        raise RuntimeError(
            f"Missing {len(missing_required)} required real captions. "
            f"See coverage report. Fill them manually rather than fake prompts."
        )


if __name__=="__main__":
    main()
