"""Repair optional TorchAO/Diffusers imports in a Colab environment.

H3 RGBA VAE training does not use TorchAO quantization. Some Colab images
ship TorchAO versions without `torchao.quantization.FqnToConfig`, which
causes Diffusers' model loader to fail before testing or training.

Only uninstall TorchAO when it is present but cannot supply the API expected
by the pinned Diffusers version. Run this *before* importing Diffusers.

Usage: python tools/colab_compat.py --repair
"""
import argparse
import importlib.metadata
import subprocess
import sys


CHECK = (
    "from torchao.quantization import FqnToConfig, quantize_; "
    "print('TORCHAO_API_OK')"
)


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def check_torchao():
    if package_version("torchao") is None:
        print("TORCHAO_NOT_INSTALLED: nothing to repair", flush=True)
        return True
    result = subprocess.run(
        [sys.executable, "-c", CHECK], text=True, capture_output=True
    )
    if result.returncode == 0:
        print("TORCHAO_COMPATIBLE: " + package_version("torchao"), flush=True)
        return True
    print("TORCHAO_INCOMPATIBLE: installed version = " + package_version("torchao"))
    print(result.stderr[-3500:], flush=True)
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repair", action="store_true",
                        help="Remove broken optional TorchAO package")
    args = parser.parse_args()
    valid = check_torchao()
    if not valid and args.repair:
        print("Removing TorchAO; this VAE workload does not require it.", flush=True)
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
            check=True,
        )
        valid = check_torchao()
        print("RESTART_NOTICE: if TorchAO or Diffusers was imported in this "
              "notebook kernel, restart the runtime before importing it again. "
              "The standalone training script launches in a clean process.",
              flush=True)
    if not valid:
        raise SystemExit(
            "Incompatible optional torchao. Rerun with --repair before importing Diffusers."
        )
    probe = subprocess.run(
        [sys.executable, "-c",
         "from diffusers import AutoencoderKLMiniMaxH3; "
         "print('H3_DIFFUSERS_IMPORT_OK')"],
        capture_output=True, text=True,
    )
    print(probe.stdout, end="", flush=True)
    if probe.returncode:
        print(probe.stderr[-7000:], flush=True)
        raise SystemExit(probe.returncode)


if __name__ == "__main__":
    main()
