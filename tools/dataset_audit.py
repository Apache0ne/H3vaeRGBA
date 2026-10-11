"""Audit paired foreground/alpha video directories without a GPU.

Expected inputs: extracted data_fgr and data_pha archives. They may contain
nested directories; clip matching is by relative path with known role tags
stripped from filenames, or by unique stem when directories differ.
"""
import argparse
import json
import pathlib
import re

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}


def key(path, root):
    relative = path.relative_to(root).with_suffix("").as_posix().lower()
    parts = [p for p in relative.split("/") if p not in
             ("data_fgr", "data_pha", "foreground", "alpha", "fgr", "pha", "videos")]
    candidate = "/".join(parts)
    candidate = re.sub(r"(?i)([-_](?:fgr|foreground|pha|alpha|matte))+$", "", candidate)
    return candidate


def discover(root):
    root = pathlib.Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Directory not found: {root}. Extract the .zip archive first.")
    indexed = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
            k = key(path, root)
            if k in indexed:
                raise ValueError(f"Duplicate key '{k}': {indexed[k]} and {path}")
            indexed[k] = path
    return indexed


def audit_pair(foreground, alpha, min_frames=17):
    import cv2
    probes = []
    for path in (foreground, alpha):
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"Failed to decode: {path}")
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        good, frame = cap.read()
        cap.release()
        if not good or frame is None:
            raise ValueError(f"Unreadable first frame: {path}")
        probes.append(dict(frames=n, width=w, height=h, fps=fps))
    f, a = probes
    if f["frames"] < min_frames or a["frames"] < min_frames:
        raise ValueError(f"Insufficient frames: {f['frames']} / {a['frames']}")
    if f["width"] != a["width"] or f["height"] != a["height"]:
        raise ValueError("Foreground/alpha frame dimensions differ")
    if abs(f["fps"] - a["fps"]) > 0.1:
        raise ValueError("Foreground/alpha FPS differs")
    if abs(f["frames"] - a["frames"]) > 1:
        raise ValueError("Foreground/alpha frame count differs")
    return f


def build_manifest(foreground_root, alpha_root, output, min_frames=17):
    fg, alpha = discover(foreground_root), discover(alpha_root)
    if not fg or not alpha:
        raise ValueError("No supported videos. Confirm .zip files have been extracted.")
    common = set(fg) & set(alpha)
    # Flatten matching only if one or both datasets use different nesting
    if not common:
        def flatten(d):
            ret = {}
            for k, p in d.items():
                short = k.split("/")[-1]
                if short in ret:
                    raise ValueError(f"Ambiguous flattened stem: {short}")
                ret[short] = p
            return ret
        fg, alpha = flatten(fg), flatten(alpha)
        common = set(fg) & set(alpha)
    rows, errors = [], []
    for k in sorted(common):
        try:
            info = audit_pair(fg[k], alpha[k], min_frames)
            rows.append(dict(id=k, foreground=str(fg[k]), alpha=str(alpha[k]), **info))
        except Exception as e:
            errors.append(dict(id=k, error=str(e)))
    report = dict(clips=rows, rejected=errors,
                  unmatched_foreground=sorted(set(fg) - set(alpha)),
                  unmatched_alpha=sorted(set(alpha) - set(fg)),
                  stats=dict(foreground=len(fg), alpha=len(alpha), valid=len(rows),
                             invalid=len(errors), unmatched=len(set(fg) ^ set(alpha))))
    pathlib.Path(output).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report["stats"], indent=2))
    if not rows:
        raise RuntimeError("No valid foreground+alpha pairs. Inspect the report.")
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--foreground", required=True)
    p.add_argument("--alpha", required=True)
    p.add_argument("--output", default="manifest.json")
    p.add_argument("--min-frames", type=int, default=17)
    args = p.parse_args()
    build_manifest(args.foreground, args.alpha, args.output, args.min_frames)


if __name__ == "__main__":
    main()
