"""CPU-only download, safe extraction and pairing audit for Wan-Alpha data."""
import argparse
from pathlib import Path
import stat
import zipfile

from dataset_audit import build_manifest


def extract_safely(archive, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for entry in z.infolist():
            target = (destination / entry.filename).resolve()
            if not target.is_relative_to(destination):
                raise RuntimeError(f"Unsafe archive path: {entry.filename}")
            mode = (entry.external_attr >> 16) & 0xffff
            if stat.S_ISLNK(mode):
                raise RuntimeError(f"Symlink in archive: {entry.filename}")
        z.extractall(destination)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workdir", default="./wan_alpha_data")
    p.add_argument("--skip-download", action="store_true")
    args = p.parse_args()
    root = Path(args.workdir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    downloads = root / "downloads"
    if not args.skip_download:
        from huggingface_hub import snapshot_download
        snapshot_download(
            "htdong/Wan-Alpha-dataset",
            repo_type="dataset",
            allow_patterns=["data_fgr.zip", "data_pha.zip"],
            local_dir=str(downloads),
        )
    fgr_zip = downloads / "data_fgr.zip"
    pha_zip = downloads / "data_pha.zip"
    if not fgr_zip.is_file() or not pha_zip.is_file():
        raise FileNotFoundError("Both archives are required in workdir/downloads")
    foreground, alpha = root / "foreground", root / "alpha"
    for archive, dest in ((fgr_zip, foreground), (pha_zip, alpha)):
        print(f"Extracting {archive.name} ...", flush=True)
        extract_safely(archive, dest)
    build_manifest(foreground, alpha, root / "manifest.json", min_frames=17)
    print("Manifest:", root / "manifest.json")


if __name__ == "__main__":
    main()
