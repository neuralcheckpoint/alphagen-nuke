"""Medratools for Nuke: install Alpha Gen, start its ComfyUI backend, download the LTX-2.5 weights.

Run through medratools.sh (Linux) or medratools.bat (Windows); they provide uv.
Standard library only, so the same steps run on both systems.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
ENGINE = HERE / "AlphaGen"
AITOOLS = HERE / "aitools"
if not AITOOLS.is_dir():
    AITOOLS = HERE.parent / "aitools"
SETTINGS = HERE / "medratools.json"
WINDOWS = os.name == "nt"

sys.path.insert(0, str(AITOOLS))
from aitools_engine.i18n import tr  # noqa: E402  (standard library only)

# Backend pinned to the versions Alpha Gen was validated with.
COMFYUI = ("comfyanonymous/ComfyUI", "20ca544ee0436721d8eb5f544665e490609f72c8")
LTXVIDEO = ("Lightricks/ComfyUI-LTXVideo", "dfb2786749af36f200ea023388dc729a3e106b42")
TORCH = ["torch==2.14.0", "torchvision==0.29.0", "torchaudio==2.11.0"]
TORCH_INDEX = "https://download.pytorch.org/whl/cu130"
# kornia 0.8.3 dropped the pyramid `pad` that ComfyUI-LTXVideo at LTXVIDEO imports.
KORNIA = "kornia==0.8.2"
HF = "huggingface_hub==1.32.0"
SAM_TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
SAM_CHECKPOINTS = [
    (
        "sam2.1_hiera_large.pt", 898083611,
        "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318",
    ),
    (
        "sam2.1_hiera_base_plus.pt", 323606802,
        "a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5",
    ),
]
SAM_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/{}"

LTX = ("Lightricks/LTX-2.5", "2356ce76915d6c48d313d7e8b25900e1dd3abaa8")
ALPHA = ("Lightricks/LTX-2.5-22b-IC-LoRA-Alpha-Gen", "6184df14b1b560bf9b6447d3ee126bd7e4a88513")
# (repo, file in repo, ComfyUI folder the repo is mirrored into, bytes)
MODELS = [
    (LTX, "diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors", "models", 21504034224),
    (LTX, "text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors", "models", 15372969374),
    (LTX, "vae/ltx-2.5-video-vae-bf16.safetensors", "models", 1472223346),
    (LTX, "vae/ltx-2.5-audio-vae-bf16.safetensors", "models", 364866540),
    (ALPHA, "ltx-2.5-22b-ic-lora-alpha-gen-0.9.safetensors", "models/loras", 1308787472),
]
LICENSE = "https://github.com/Lightricks/LTX-2/blob/main/LICENSE-2_x"


def _uv():
    path = os.environ.get("UV") or shutil.which("uv")
    if not path:
        sys.exit("uv not found: run this through medratools.sh or medratools.bat")
    return path


def _python(venv):
    return venv / ("Scripts/python.exe" if WINDOWS else "bin/python")


def _run(*command, cwd=None):
    print("+", " ".join(str(part) for part in command), flush=True)
    subprocess.run([str(part) for part in command], cwd=cwd, check=True)


def _settings():
    if SETTINGS.is_file():
        return json.loads(SETTINGS.read_text(encoding="utf-8"))
    return {}


def _rototrack_root():
    for root in (HERE / "RotoTrack", HERE.parent / "RotoTrack"):
        if (root / "rototrack").is_dir():
            return root
    sys.exit(tr("No RotoTrack engine found beside Medratools"))


def _comfy_root():
    root = _settings().get("comfyui_root")
    if not root:
        sys.exit(tr("Medratools is not installed yet: run install first"))
    return Path(root)


def _fetch(repo, commit, destination):
    url = "https://codeload.github.com/{}/zip/{}".format(repo, commit)
    print("+ download", url, flush=True)
    with tempfile.TemporaryDirectory(prefix="medratools-", dir=destination.parent) as temp:
        archive = Path(temp) / "source.zip"
        with urlopen(url, timeout=60) as response, open(archive, "wb") as handle:
            shutil.copyfileobj(response, handle)
        with zipfile.ZipFile(archive) as source:
            source.extractall(temp)
        (top,) = [path for path in Path(temp).iterdir() if path.is_dir()]
        top.rename(destination)


def install_backend(root):
    if root.exists() and any(root.iterdir()):
        sys.exit(tr("{} is not empty and is not a ComfyUI folder", root))
    root.parent.mkdir(parents=True, exist_ok=True)
    if root.exists():
        root.rmdir()
    _fetch(*COMFYUI, root)
    _fetch(*LTXVIDEO, root / "custom_nodes" / "ComfyUI-LTXVideo")
    uv, python = _uv(), _python(root / ".venv")
    _run(uv, "venv", "--python", "3.13", root / ".venv")
    _run(uv, "pip", "install", "--python", python, "--index-url", TORCH_INDEX, *TORCH)
    _run(uv, "pip", "install", "--python", python, "-r", root / "requirements.txt",
         "-r", root / "custom_nodes" / "ComfyUI-LTXVideo" / "requirements.txt", KORNIA)


def register_nuke():
    """Add the plugin to ~/.nuke/init.py; lines tagged '# Medratools' are rewritten on reinstall."""
    init = Path.home() / ".nuke" / "init.py"
    init.parent.mkdir(parents=True, exist_ok=True)
    lines = init.read_text(encoding="utf-8").splitlines() if init.is_file() else []
    lines = [line for line in lines if not line.endswith("# Medratools")]
    lines += [
        "import nuke  # Medratools",
        "nuke.pluginAddPath({!r})  # Medratools".format((ENGINE / "nuke_plugin").as_posix()),
    ]
    init.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(tr("Registered the Medratools menu in"), init)


def install(args):
    uv, python = _uv(), _python(ENGINE / ".venv")
    if not python.is_file():
        _run(uv, "venv", "--python", "3.12", ENGINE / ".venv")
    # The engine only resizes and encodes frames; inference runs in ComfyUI.
    _run(uv, "pip", "install", "--python", python, "--index-url", "https://download.pytorch.org/whl/cpu", "torch")
    _run(uv, "pip", "install", "--python", python, "-r", ENGINE / "requirements.txt", AITOOLS)

    root = Path(args.comfyui).expanduser().resolve() if args.comfyui else HERE / "ComfyUI"
    if (root / "main.py").is_file():
        print(tr("Using the existing ComfyUI in"), root)
        if not (root / "custom_nodes" / "ComfyUI-LTXVideo").is_dir():
            print(tr("Warning: custom_nodes/ComfyUI-LTXVideo is missing; Alpha Gen needs it"))
    else:
        install_backend(root)
    SETTINGS.write_text(json.dumps({"comfyui_root": str(root), "port": args.port}, indent=2) + "\n", encoding="utf-8")
    register_nuke()
    print(tr("\nNext: download the models, start the backend, then open Nuke > Medratools > Alpha Gen."))


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_sam_checkpoint(models, filename, expected_size, expected_sha256):
    target = models / filename
    if target.is_file() and target.stat().st_size == expected_size and _sha256(target) == expected_sha256:
        print(tr("SAM 2.1 checkpoint already verified:"), target)
        return

    models.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    digest = hashlib.sha256()
    received = 0
    url = SAM_CHECKPOINT_URL.format(filename)
    print(tr("Downloading the official Apache-2.0 SAM 2.1 checkpoint:"), url, flush=True)
    try:
        with urlopen(url, timeout=60) as response, open(partial, "wb") as handle:
            while True:
                block = response.read(8 * 1024 * 1024)
                if not block:
                    break
                handle.write(block)
                digest.update(block)
                received += len(block)
            handle.flush()
            os.fsync(handle.fileno())
        if received != expected_size or digest.hexdigest() != expected_sha256:
            raise ValueError(tr("SAM 2.1 checkpoint integrity check failed for {}", filename))
        os.replace(partial, target)
    except Exception:
        try:
            partial.unlink()
        except OSError:
            pass
        raise
    print(tr("Verified SAM 2.1 checkpoint:"), target, received, "bytes", digest.hexdigest())


def sam(args):
    root = _rototrack_root()
    config = root / "vendor" / "sam2" / "sam2" / "configs" / "sam2.1" / "sam2.1_hiera_l.yaml"
    if not config.is_file():
        sys.exit(tr("SAM 2.1 source/config files are missing from {}; reinstall the package", root))

    python = _python(root / ".venv")
    if python.is_file():
        print(tr("Reusing the existing SAM environment at"), python.parent.parent)
    else:
        uv = _uv()
        _run(uv, "venv", "--python", "3.12", root / ".venv")
        _run(uv, "pip", "install", "--python", python, "--index-url", SAM_TORCH_INDEX,
             "torch", "torchvision")
        _run(uv, "pip", "install", "--python", python, "-r", root / "requirements.txt", AITOOLS)

    models = root / "models"
    for filename, size, sha256 in SAM_CHECKPOINTS:
        _download_sam_checkpoint(models, filename, size, sha256)
    print(tr("SAM 2.1 subject selection is ready in the separate GPU environment at"), root)


def backend(args):
    root = _comfy_root()
    python = _python(root / ".venv")
    portable = root.parent / "python_embeded" / "python.exe"  # ComfyUI Windows portable layout
    if not python.is_file() and portable.is_file():
        python = portable
    port = str(_settings().get("port", 8188))
    if not python.is_file():
        sys.exit(tr("No Python found for {}; start that ComfyUI yourself on port {}", root, port))
    print(tr("ComfyUI for Alpha Gen on http://127.0.0.1:{} (Ctrl+C stops it)", port), flush=True)
    try:
        return subprocess.call([str(python), "main.py", "--listen", "127.0.0.1", "--port", port], cwd=root)
    except KeyboardInterrupt:
        return 0


def models(args):
    root = _comfy_root()
    hf = [_uv(), "tool", "run", "--from", HF, "hf"]
    missing = {}
    for repo, path, folder, size in MODELS:
        target = root / folder / path
        if not target.is_file() or target.stat().st_size != size:
            missing.setdefault((repo, folder), []).append(path)
    if not missing:
        print(tr("All Alpha Gen models are in"), root / "models")
        return 0
    print(tr("The LTX-2.5 weights are licensed under the LTX-2 Community License:"), LICENSE)
    print(tr("Accept the terms with your Hugging Face account first:"))
    for name in sorted({repo[0] for repo, _ in missing}):
        print("  https://huggingface.co/" + name)
    if subprocess.run(hf + ["auth", "whoami"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
        _run(*hf, "auth", "login")
    for ((name, revision), folder), paths in missing.items():
        _run(*hf, "download", name, *paths, "--revision", revision, "--local-dir", root / folder)
    return 0


def check(args):
    return subprocess.call([str(_python(ENGINE / ".venv")), "-m", "alphagen", "--check"], cwd=ENGINE)


def main():
    parser = argparse.ArgumentParser(prog="medratools", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("install", help="engine, ComfyUI backend and Nuke menu")
    setup.add_argument("--comfyui", help="existing ComfyUI folder to reuse, or where to install it")
    setup.add_argument("--port", type=int, default=8188)
    commands.add_parser("models", help="download the LTX-2.5 weights (~40 GB) from Hugging Face")
    commands.add_parser("backend", help="start ComfyUI for Alpha Gen")
    commands.add_parser("check", help="verify models, backend and nodes (backend must be running)")
    commands.add_parser("sam", help=tr("install the optional SAM 2.1 subject-selection environment and weights"))
    args = parser.parse_args()
    return {
        "install": install, "models": models, "backend": backend, "check": check, "sam": sam,
    }[args.command](args) or 0


if __name__ == "__main__":
    sys.exit(main())
