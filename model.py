import importlib
import importlib.abc
import importlib.machinery
import os
import sys
import types
import warnings
from pathlib import Path

import torch

# Stub speechbrain optional integrations (numba, k2_fsa) to prevent import errors on Windows
# These are optional dependencies that may fail to import but are not required for inference


class SpeechbrainIntegrationStubLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return types.ModuleType(spec.name)

    def exec_module(self, module):
        spec = module.__spec__
        module.__file__ = "<stub>"
        module.__package__ = spec.name if spec.submodule_search_locations is not None else spec.name.rpartition(".")[0]
        if spec.submodule_search_locations is not None:
            module.__path__ = []
        module.__all__ = []


class SpeechbrainIntegrationStubFinder(importlib.abc.MetaPathFinder):
    NAMESPACE = "speechbrain.integrations"

    def find_spec(self, fullname, path, target=None):
        if not fullname.startswith(self.NAMESPACE):
            return None
        if fullname in sys.modules:
            return None

        real_spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if real_spec is not None:
            return None

        is_pkg = "." not in fullname
        spec = importlib.machinery.ModuleSpec(fullname, SpeechbrainIntegrationStubLoader(), is_package=is_pkg)
        if is_pkg:
            spec.submodule_search_locations = []
        sys.modules[fullname] = types.ModuleType(fullname)
        sys.modules[fullname].__all__ = []
        return spec


def _install_speechbrain_optional_integration_stub_finder():
    if not any(isinstance(finder, SpeechbrainIntegrationStubFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, SpeechbrainIntegrationStubFinder())
    if "speechbrain.integrations" not in sys.modules:
        base_mod = types.ModuleType("speechbrain.integrations")
        base_mod.__path__ = []
        sys.modules["speechbrain.integrations"] = base_mod
    for submodule in ["huggingface", "numba", "k2_fsa", "nlp"]:
        fullname = f"speechbrain.integrations.{submodule}"
        if fullname not in sys.modules:
            submod = types.ModuleType(fullname)
            submod.__path__ = []
            submod.__package__ = "speechbrain.integrations"
            submod.__all__ = []
            sys.modules[fullname] = submod


_install_speechbrain_optional_integration_stub_finder()


SEPFORNER_MODEL_SOURCE = os.environ.get("SEPFORNER_MODEL_SOURCE", "speechbrain/sepformer-libri3mix")
SEPFORNER_MODEL_REVISION = os.environ.get("SEPFORNER_MODEL_REVISION", "main")
SEPFORNER_REQUIRED_FILES = ("hyperparams.yaml", "encoder.ckpt", "decoder.ckpt", "masknet.ckpt")


def _local_sepformer_dir() -> Path:
    return Path(os.path.abspath("./pretrained_sepformer"))


def _missing_sepformer_files(local_dir: Path):
    return [
        filename
        for filename in SEPFORNER_REQUIRED_FILES
        if not (local_dir / filename).is_file() or (local_dir / filename).stat().st_size == 0
    ]


def _download_missing_sepformer_files(local_dir: Path) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "huggingface_hub is required to download SepFormer assets. Install it with `pip install huggingface_hub`."
        ) from exc

    print(f"SepFormer source: {SEPFORNER_MODEL_SOURCE}@{SEPFORNER_MODEL_REVISION}")
    for filename in SEPFORNER_REQUIRED_FILES:
        local_path = local_dir / filename
        is_file = local_path.is_file()
        file_size = local_path.stat().st_size if is_file else 0

        if is_file and file_size > 0:
            print(f"Using existing SepFormer asset: {local_path}")
            continue

        status_msg = "missing" if not is_file else "empty"
        print(f"Local asset '{filename}' is {status_msg}. Downloading from '{SEPFORNER_MODEL_SOURCE}' to '{local_dir}'...")

        try:
            hf_hub_download(
                repo_id=SEPFORNER_MODEL_SOURCE,
                filename=filename,
                revision=SEPFORNER_MODEL_REVISION,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download '{filename}' from '{SEPFORNER_MODEL_SOURCE}' revision '{SEPFORNER_MODEL_REVISION}'. "
                "Check your network connection or set SEPFORNER_MODEL_SOURCE/SEPFORNER_MODEL_REVISION to a reachable SpeechBrain model."
            ) from exc


def ensure_local_sepformer_assets() -> Path:
    local_dir = _local_sepformer_dir()
    missing = _missing_sepformer_files(local_dir)
    if missing:
        _download_missing_sepformer_files(local_dir)

    missing = _missing_sepformer_files(local_dir)
    if missing:
        raise FileNotFoundError(
            f"Local pretrained SepFormer directory '{local_dir}' is missing required files: {missing}. "
            f"Set SEPFORNER_MODEL_SOURCE to a valid SpeechBrain SepFormer model and rerun the application."
        )

    return local_dir


class UnifiedSepFormer(torch.nn.Module):
    def __init__(self, modules_dict):
        super().__init__()

        self.encoder = modules_dict['encoder']
        self.masknet = modules_dict['masknet']
        self.decoder = modules_dict['decoder']

    def forward(self, mix):
        mix_w = self.encoder(mix)
        est_mask = self.masknet(mix_w)

        decoded_sources = []

        for i in range(est_mask.shape[0]):
            sep_h_i = mix_w * est_mask[i]
            est_source_i = self.decoder(sep_h_i)
            decoded_sources.append(est_source_i.unsqueeze(-1))

        est_source = torch.cat(decoded_sources, dim=-1)
        return est_source


def load_model(checkpoint_path=None):
    try:
        speechbrain_inference = importlib.import_module("speechbrain.inference.separation")
        speechbrain_fetching = importlib.import_module("speechbrain.utils.fetching")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "SpeechBrain is required for SepFormer model loading. Install it with `pip install speechbrain` and a compatible `k2` package, or use a separate environment where SpeechBrain is supported."
        ) from exc

    _install_speechbrain_optional_integration_stub_finder()
    SepformerSeparation = getattr(speechbrain_inference, "SepformerSeparation")
    LocalStrategy = getattr(speechbrain_fetching, "LocalStrategy")

    local_sepformer_dir = ensure_local_sepformer_assets()

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            model_hub = SepformerSeparation.from_hparams(
                source=str(local_sepformer_dir),
                savedir=str(local_sepformer_dir),
                local_strategy=LocalStrategy.COPY_SKIP_CACHE,
            )
    except ImportError as exc:
        msg = str(exc)
        if "speechbrain.integrations.k2_fsa" in msg or "Please install k2 to use k2" in msg or "No module named '_k2'" in msg:
            raise ImportError(
                "SpeechBrain attempted to load the optional k2 integration and failed. "
                "This often happens on Windows because k2 is not available or the installed wheel is incompatible. "
                "If you do not need k2 features, use a SpeechBrain install that does not require k2 or run this project on Linux. "
                "Original error: " + msg
            ) from exc
        raise

    model = UnifiedSepFormer(model_hub.mods)

    if checkpoint_path is None:
        model.eval()
        return model

    if not os.path.exists(checkpoint_path):
        print(f"WARNING: checkpoint '{checkpoint_path}' not found. Using local pretrained model instead.")
        model.eval()
        return model

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"TypedStorage is deprecated.*")
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu"
        )

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as err:
        print("WARNING: checkpoint is incompatible with the local SepFormer architecture.")
        print("Attempting relaxed load with strict=False.")
        try:
            load_result = model.load_state_dict(state_dict, strict=False)
            missing = getattr(load_result, "missing_keys", None)
            unexpected = getattr(load_result, "unexpected_keys", None)
            if missing:
                print("Missing keys from checkpoint:", missing)
            if unexpected:
                print("Unexpected keys in checkpoint:", unexpected)
            print("Relaxed checkpoint load succeeded. Using loaded weights where possible.")
            model.eval()
            return model
        except RuntimeError as err2:
            print("Relaxed checkpoint load also failed. Using local pretrained SepFormer weights from './pretrained_sepformer' instead.")
            print(err2)
            model.eval()
            return model

    model.eval()
    return model