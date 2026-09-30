"""Photon CT/MR inference using the archived Challenge compute path."""

from __future__ import annotations
import json
import math
import hashlib
import os
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from itertools import product
from pathlib import Path
import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts"))
from prismx.aperture import MLCAperture
from prismx.config import MLCSpec
from prismx.geometry import BeamGeometry
from prismx.hu import ct_volume_from_hu
from prismx.mr_data import normalize_mr
from prismx.mr_density import (
    TRAIN_SPACING_MM,
    density_ctvolume,
    needs_resample,
    resample_like,
    resample_to_spacing,
)

PHOTON_SCALE = 5.437449455848826e-05
PHOTON_MRI_SCALE_JSON = _REPO / "commissioning" / "photon_mri_dose_scale.json"
_REGION_TO_SITE = {"thoracic": "thx", "abdominal": "abd"}
N_INPUT_SLOTS = 10
N_OUTPUT_SLOTS = 10
OUTPUT_IMAGE_ROOT = "images"
OUTPUT_FILENAME = "output.mha"
PHOTON_INFERENCE_REGION = "body_filled"
MR_DENSITY_MAX_TILE_VOXELS = 12000000
MR_DENSITY_TILE_OVERLAP = 0.75


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class _CutoffDiagnostics:
    """Aggregate cutoff impact without retaining any dose maps."""

    def __init__(self):
        self.rows: list[tuple[float, float, float, float, float, float]] = []

    def prepare(self, dose: torch.Tensor, cutoff: float) -> torch.Tensor:
        positive = dose > 0.0
        removed = positive & (dose <= float(cutoff))
        return torch.stack(
            (
                dose.max().to(torch.float64),
                positive.sum().to(torch.float64),
                removed.sum().to(torch.float64),
                dose.clamp_min(0.0).sum(dtype=torch.float64),
                dose.masked_select(removed).sum(dtype=torch.float64),
            )
        )

    def add(self, cutoff: float, packed: torch.Tensor) -> None:
        maximum, n_positive, n_removed, total, removed_total = (
            float(v) for v in packed.detach().cpu().tolist()
        )
        self.rows.append(
            (float(cutoff), maximum, n_positive, n_removed, total, removed_total)
        )

    def report(self) -> None:
        if not self.rows:
            return
        arr = np.asarray(self.rows, dtype=np.float64)
        ratios = np.divide(
            arr[:, 0], arr[:, 1], out=np.full(len(arr), np.nan), where=arr[:, 1] > 0
        )
        finite = ratios[np.isfinite(ratios)]
        ratio_text = (
            f"{np.min(finite):.3e}/{np.median(finite):.3e}/{np.max(finite):.3e}"
            if finite.size
            else "n/a"
        )
        positive = float(arr[:, 2].sum())
        removed = float(arr[:, 3].sum())
        total = float(arr[:, 4].sum())
        removed_total = float(arr[:, 5].sum())
        print(
            "[cutoff] maps={} cutoff/max min/median/max={} removed-positive={:.6%} removed-dose-integral={:.6%}".format(
                len(arr),
                ratio_text,
                removed / positive if positive else 0.0,
                removed_total / total if total else 0.0,
            ),
            flush=True,
        )


def _apply_minimum_cutoff_(
    dose: torch.Tensor, cutoff: float, diagnostics: _CutoffDiagnostics | None = None
) -> torch.Tensor | None:
    """Conservatively enforce that no positive output is below ``cutoff``."""
    packed = diagnostics.prepare(dose, cutoff) if diagnostics is not None else None
    dose.masked_fill_(dose <= float(cutoff), 0.0)
    return packed


def _read_image(mha_path: Path):
    """Read a .mha -> (array [Z,Y,X] float32, spacing (sx,sy,sz), origin, sitk img)."""
    import SimpleITK as sitk

    img = sitk.ReadImage(str(mha_path))
    d = img.GetDirection()
    ident = (1, 0, 0, 0, 1, 0, 0, 0, 1)
    if max((abs(a - b) for a, b in zip(d, ident))) > 0.0001:
        raise ValueError(
            f"{mha_path}: non-identity image direction {d}; the engine assumes world = origin + spacing*index. Handle direction first."
        )
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    return (arr, tuple(img.GetSpacing()), tuple(img.GetOrigin()), img)


def _find_mha(slot_dir: Path) -> Path:
    """The single .mha inside a source-image slot directory."""
    mhas = sorted(slot_dir.glob("*.mha"))
    if len(mhas) != 1:
        raise ValueError(f"expected exactly one .mha in {slot_dir}, found {len(mhas)}")
    return mhas[0]


def _per_site_photon_scale(site: str) -> float:
    tbl = json.loads(PHOTON_MRI_SCALE_JSON.read_text())["dose_scale"]
    v = tbl.get(site)
    if v is None or float(v) <= 0:
        raise SystemExit(
            f"no committed photon_mri scale for site {site!r} in {PHOTON_MRI_SCALE_JSON}"
        )
    return float(v)


def _trace_photon_decoders(engine, device: torch.device) -> float:
    """Trace/freeze the two shape-dynamic inference-only Photon MLPs."""
    started = time.perf_counter()
    cfg = engine.cfg.decoder
    main_example = (
        torch.zeros(8, cfg.latent_channels, device=device),
        torch.zeros(8, cfg.n_analytic, device=device),
    )
    far_example = (torch.zeros(8, cfg.n_analytic, device=device),)
    engine.main_decoder = torch.jit.freeze(
        torch.jit.trace(engine.main_decoder.eval(), main_example, strict=True)
    )
    engine.far_decoder = torch.jit.freeze(
        torch.jit.trace(engine.far_decoder.eval(), far_example, strict=True)
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter() - started


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _commissioning_sha256(directory: Path) -> str:
    """Digest the canonical commissioned-table manifest used at build time."""
    directory = Path(directory)
    lines = []
    for path in sorted(
        (entry for entry in directory.rglob("*") if entry.is_file()),
        key=lambda entry: entry.relative_to(directory).as_posix(),
    ):
        relative = (Path("commissioning") / path.relative_to(directory)).as_posix()
        lines.append(f"{_sha256_file(path)}  {relative}\n")
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def _verify_shipped_weights(cfg: dict, image_type: str, modality: str) -> None:
    """Refuse to serve if the weights on disk are not the ones we gated."""
    if _env_flag("PRISMX_SKIP_WEIGHT_VERIFY", default=False):
        print("[verify] SKIPPED by PRISMX_SKIP_WEIGHT_VERIFY", flush=True)
        return
    checks = []
    engine_key = "engine"
    checks.append(("engine", cfg.get(engine_key), "PRISMX_EXPECT_ENGINE_SHA256"))
    if image_type == "mri":
        checks.append(("mr-density-net", cfg.get("net"), "PRISMX_EXPECT_MR_NET_SHA256"))
    problems = []
    for name, path, env_key in checks:
        expected = os.environ.get(env_key, "")
        if not expected or expected in ("unknown", "none"):
            problems.append(f"{name}: no expected digest baked into the image")
            continue
        if path is None:
            problems.append(f"{name}: no path configured")
            continue
        actual = _sha256_file(Path(path))
        state = "ok" if actual == expected else "MISMATCH"
        print(f"[verify] {name} sha256 {actual[:16]} {state}", flush=True)
        if actual != expected:
            problems.append(f"{name}: expected {expected[:16]}, found {actual[:16]}")
    expected_comm = os.environ.get("PRISMX_EXPECT_COMMISSIONING_SHA256", "")
    comm_dir = _REPO / "commissioning"
    if not expected_comm or expected_comm == "unknown":
        problems.append("commissioning: no expected digest baked into the image")
    elif not comm_dir.is_dir():
        problems.append("commissioning: directory is missing")
    else:
        actual_comm = _commissioning_sha256(comm_dir)
        state = "ok" if actual_comm == expected_comm else "MISMATCH"
        print(f"[verify] commissioning sha256 {actual_comm[:16]} {state}", flush=True)
        if actual_comm != expected_comm:
            problems.append(
                f"commissioning: expected {expected_comm[:16]}, found {actual_comm[:16]}"
            )
    print(f"[verify] git rev {os.environ.get('PRISMX_GIT_REV', 'unknown')}", flush=True)
    if problems:
        raise SystemExit(
            "[verify] REFUSING TO SERVE, shipped weights are not the gated ones: "
            + "; ".join(problems)
        )


class Models:
    """Frozen Photon engine and optional MR density network."""

    def __init__(self, modality: str, image_type: str, device, cfg: dict):
        if modality != "photon":
            raise ValueError("This release supports Photon only")
        _verify_shipped_weights(cfg, image_type, modality)
        self.modality = modality
        self.image_type = image_type
        self.device = torch.device(device)
        from prismx import loaders as tmr

        self.engine = tmr.load_frozen_engine(
            Path(cfg["engine"]), Path(cfg["pdd"]), self.device
        )
        from dataclasses import replace as _replace

        edt_stride = int(os.environ.get("PRISMX_EDT_STRIDE", "1"))
        scout_stride = int(os.environ.get("PRISMX_WINDOW_SCOUT_STRIDE", "1"))
        if edt_stride > 1 or scout_stride > 1:
            self.engine.cfg = _replace(
                self.engine.cfg,
                raster=_replace(self.engine.cfg.raster, edt_stride=edt_stride),
                cone=_replace(self.engine.cfg.cone, window_scout_stride=scout_stride),
            )
            print(
                f"[runtime-gates] edt_stride={edt_stride} window_scout_stride={scout_stride}",
                flush=True,
            )
        if _env_flag("PRISMX_TRACE_PHOTON_DECODERS", default=True):
            trace_s = _trace_photon_decoders(self.engine, self.device)
            print(
                f"[runtime-gates] traced photon decoders in {trace_s:.3f}s", flush=True
            )
        self.pscale_const = PHOTON_SCALE
        self.net = None
        if image_type == "mri":
            from prismx.loaders import load_density_net

            self.net = load_density_net(Path(cfg["net"]), self.device)
            if self.device.type == "cuda":
                self.net.half()
        self.engine.eval().requires_grad_(False)
        if self.net is not None:
            self.net.eval().requires_grad_(False)

    def scale(self, site: str) -> float:
        """Raw-unit dose scale (pred * scale = raw MC units, matching the GT)."""
        if self.image_type == "mri":
            return _per_site_photon_scale(site)
        return self.pscale_const


def _tile_starts(length: int, window: int, overlap: float) -> list[int]:
    """Cover one axis with fixed windows, including an exact final edge window."""
    length, window = (int(length), min(int(window), int(length)))
    if window >= length:
        return [0]
    step = max(1, int(round(window * (1.0 - float(overlap)))))
    starts = list(range(0, length - window + 1, step))
    final = length - window
    if starts[-1] != final:
        starts.append(final)
    return starts


def _mr_tile_shape(
    shape: tuple[int, int, int], max_voxels: int, mult: int
) -> tuple[int, int, int]:
    """Reduce as few spatial axes as possible to fit a density-forward budget."""
    tile = [int(v) for v in shape]
    max_voxels = max(int(max_voxels), int(mult) ** 3)
    while math.prod(tile) > max_voxels:
        axis = max(range(3), key=lambda i: tile[i])
        other = math.prod(tile) // tile[axis]
        target = max(mult, max_voxels // max(1, other) // mult * mult)
        if target >= tile[axis]:
            target = max(mult, tile[axis] // 2 // mult * mult)
        tile[axis] = target
    return tuple((min(s, t) for s, t in zip(shape, tile)))


def _mr_blend_weight(
    shape: tuple[int, int, int], tiled_axes: tuple[bool, bool, bool], device
) -> torch.Tensor:
    """Smooth positive tile weight; division restores exact single-tile values."""
    weight = torch.ones(shape, dtype=torch.float32, device=device)
    for axis, is_tiled in enumerate(tiled_axes):
        if not is_tiled:
            continue
        coord = torch.linspace(
            -1.0, 1.0, shape[axis], dtype=torch.float32, device=device
        )
        one_d = torch.exp(-0.5 * (coord / 0.4).square()).clamp_min_(0.001)
        view = [1, 1, 1]
        view[axis] = shape[axis]
        weight.mul_(one_d.view(view))
    return weight


def _mr_density_forward(net, mr_norm: torch.Tensor) -> torch.Tensor:
    """One density-net forward with a float32 density result."""
    batch = mr_norm.unsqueeze(0).unsqueeze(0)
    first_parameter = next(net.parameters(), None)
    parameter_dtype = (
        first_parameter.dtype if first_parameter is not None else mr_norm.dtype
    )
    if mr_norm.device.type == "cuda" and parameter_dtype == torch.float16:
        out = net(batch.to(torch.float16))
    elif mr_norm.device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out = net(batch)
    else:
        out = net(batch.to(parameter_dtype))
    return out[0, 0].to(torch.float32)


@torch.no_grad()
def _mr_density_tiled(
    net,
    mr_norm: torch.Tensor,
    max_voxels: int,
    overlap: float = MR_DENSITY_TILE_OVERLAP,
) -> torch.Tensor:
    """Memory-bounded overlapping inference for a full ``[Z,Y,X]`` MR volume."""
    shape = tuple((int(v) for v in mr_norm.shape))
    mult = 2 ** int(getattr(net, "levels", 4))
    tile_shape = _mr_tile_shape(shape, max_voxels, mult)
    starts = [_tile_starts(n, w, overlap) for n, w in zip(shape, tile_shape)]
    tiled_axes = tuple((w < n for n, w in zip(shape, tile_shape)))
    weight = _mr_blend_weight(tile_shape, tiled_axes, mr_norm.device)
    accum = torch.zeros(shape, dtype=torch.float32, device=mr_norm.device)
    norm = torch.zeros(shape, dtype=torch.float32, device=mr_norm.device)
    for z0, y0, x0 in product(*starts):
        slices = (
            slice(z0, z0 + tile_shape[0]),
            slice(y0, y0 + tile_shape[1]),
            slice(x0, x0 + tile_shape[2]),
        )
        pred = _mr_density_forward(net, mr_norm[slices])
        accum[slices].add_(pred * weight)
        norm[slices].add_(weight)
        del pred
    print(
        f"[mr-density] tiled shape={shape} tile={tile_shape} tiles={math.prod((len(v) for v in starts))} overlap={overlap:.2f}",
        flush=True,
    )
    return accum.div_(norm.clamp_min_(1e-08))


@torch.no_grad()
def _mr_density_memory_bounded(
    net, mr_norm: torch.Tensor, *, free_bytes_override: int | None = None
) -> torch.Tensor:
    """Prefer full-volume inference; tile when the effective memory is insufficient."""
    n_voxels = int(mr_norm.numel())
    configured = int(
        os.environ.get("PRISMX_MR_MAX_TILE_VOXELS", MR_DENSITY_MAX_TILE_VOXELS)
    )
    if configured <= 0:
        configured = n_voxels
    budget = configured
    full_is_safe = True
    if mr_norm.device.type == "cuda":
        actual_free_bytes, _ = torch.cuda.mem_get_info(mr_norm.device)
        if free_bytes_override is None:
            configured_free = int(os.environ.get("PRISMX_MR_FREE_BYTES_OVERRIDE", "0"))
            free_bytes_override = configured_free if configured_free > 0 else None
        free_bytes = actual_free_bytes
        if free_bytes_override is not None:
            if int(free_bytes_override) <= 0:
                raise ValueError("free_bytes_override must be positive")
            free_bytes = min(actual_free_bytes, int(free_bytes_override))
            print(
                f"[mr-density] effective free-memory cap={free_bytes / 1024**3:.2f} GiB (actual={actual_free_bytes / 1024**3:.2f} GiB)",
                flush=True,
            )
        first_parameter = next(net.parameters(), None)
        is_half = first_parameter is not None and first_parameter.dtype == torch.float16
        bytes_per_voxel = 512.0 if is_half else 1024.0
        full_is_safe = n_voxels * bytes_per_voxel <= 0.85 * free_bytes
        safe_voxels = max(1000000, int(0.65 * free_bytes / bytes_per_voxel))
        budget = min(budget, safe_voxels)
    if full_is_safe:
        try:
            density = _mr_density_forward(net, mr_norm)
            first_parameter = next(net.parameters(), None)
            net_dtype = (
                first_parameter.dtype if first_parameter is not None else mr_norm.dtype
            )
            print(
                f"[mr-density] full-volume dtype={net_dtype} shape={tuple(mr_norm.shape)}",
                flush=True,
            )
            return density
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            budget = max(1000000, min(budget // 2, n_voxels - 1))
            print(
                f"[mr-density] full-volume CUDA OOM; retrying tiled at max_voxels={budget:,}",
                flush=True,
            )
    for attempt in range(3):
        try:
            return _mr_density_tiled(net, mr_norm, budget)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if attempt == 2:
                raise
            budget = max(1000000, budget // 2)
            print(
                f"[mr-density] tile CUDA OOM; reducing max_voxels to {budget:,}",
                flush=True,
            )
    raise AssertionError("unreachable")


def _ctvolume_for_image(models: Models, arr, spacing, origin):
    """Build the density CTVolume fed to the engine, from one input image."""
    if models.image_type == "mri":
        work, work_spacing, work_origin = (arr, spacing, origin)
        resample = needs_resample(spacing, TRAIN_SPACING_MM)
        if resample:
            work, work_spacing, work_origin = resample_to_spacing(
                arr, spacing, origin, TRAIN_SPACING_MM
            )
            print(
                f"[mr-density] {tuple(arr.shape)} @{tuple((round(float(v), 2) for v in spacing))} -> {tuple(work.shape)} @{TRAIN_SPACING_MM} for inference",
                flush=True,
            )
        t = torch.from_numpy(np.ascontiguousarray(work)).to(models.device)
        mr_norm = normalize_mr(t)
        density = _mr_density_memory_bounded(models.net, mr_norm).clamp(min=0.0)
        if resample:
            density = (
                torch.from_numpy(
                    resample_like(
                        density.float().cpu().numpy(),
                        work_spacing,
                        work_origin,
                        arr.shape,
                        spacing,
                        origin,
                    )
                )
                .clamp(min=0.0)
                .to(models.device)
            )
        return density_ctvolume(density, spacing, origin)
    return ct_volume_from_hu(torch.from_numpy(arr), spacing, origin).to(models.device)


def _filled_body_voxel_mask(density: torch.Tensor) -> torch.Tensor:
    """Fill internal axial air holes only in the sparse query mask."""
    from scipy.ndimage import binary_fill_holes

    base = density.detach().cpu().numpy() > 0.05
    filled = np.stack([binary_fill_holes(section) for section in base], axis=0)
    return torch.from_numpy(np.ascontiguousarray(filled)).to(density.device)


def _photon_aperture(beam: dict, cp: dict) -> MLCAperture:
    """MLCAperture from a submission control point (identical to PlanCase.aperture:"""
    n_pairs = int(beam["num_mlc_leaf_pairs"])
    left = torch.tensor(cp["mlc_left_int_mm"][::-1], dtype=torch.float32)
    right = torch.tensor(cp["mlc_right_int_mm"][::-1], dtype=torch.float32)
    return MLCAperture(
        left_mm=left,
        right_mm=right,
        jaw_u_mm=(-200.0, 200.0),
        jaw_v_mm=(-200.0, 200.0),
        mlc=MLCSpec(n_pairs=n_pairs),
    )


def _iter_photon(beams: list):
    """Yield (output_info, beam, cp) for every control point."""
    for beam in beams:
        for cp in beam["control_points"]:
            yield (cp["output_info"], beam, cp)


def _iter_items(modality: str, beams: list):
    if modality == "photon":
        return _iter_photon(beams)
    if modality == "proton":
        from prismx.proton_inference import iter_beamlets

        return iter_beamlets(beams)
    raise ValueError(f"Unknown modality: {modality}")


def _routing_plan(meta, modality: str) -> list[tuple[dict, dict[int, list[tuple]]]]:
    """Validate all routing before inference and return per-image, per-slot items."""
    if not isinstance(meta, list):
        raise ValueError("beam-level metadata must be a JSON array")
    seen_images: set[int] = set()
    slot_owner: dict[int, int] = {}
    plan: list[tuple[dict, dict[int, list[tuple]]]] = []
    for entry_no, entry in enumerate(meta):
        if not isinstance(entry, dict):
            raise ValueError(f"metadata entry {entry_no} is not an object")
        image_idx = int(entry["image_file_idx"])
        if not 0 <= image_idx < N_INPUT_SLOTS:
            raise ValueError(
                f"entry {entry_no}: image_file_idx {image_idx} outside 0..9"
            )
        if image_idx in seen_images:
            raise ValueError(f"image_file_idx {image_idx} occurs more than once")
        seen_images.add(image_idx)
        region = entry.get("anatomical_region")
        if region not in _REGION_TO_SITE:
            raise ValueError(
                f"entry {entry_no}: anatomical_region must be thoracic or abdominal, got {region!r}"
            )
        groups: dict[int, list[tuple]] = {}
        for item in _iter_items(modality, entry["beams"]):
            oi = item[0]
            ofi = int(oi["output_file_idx"])
            idx = int(oi["idx_in_output"])
            cutoff = float(oi["minimum_cutoff"])
            if not 0 <= ofi < N_OUTPUT_SLOTS:
                raise ValueError(
                    f"entry {entry_no}: output_file_idx {ofi} outside 0..9"
                )
            if idx < 0:
                raise ValueError(f"slot {ofi}: negative idx_in_output {idx}")
            if not math.isfinite(cutoff):
                raise ValueError(f"slot {ofi}, frame {idx}: non-finite minimum_cutoff")
            owner = slot_owner.setdefault(ofi, entry_no)
            if owner != entry_no:
                raise ValueError(
                    f"output slot {ofi} mixes input images from entries {owner} and {entry_no}"
                )
            groups.setdefault(ofi, []).append(item)
        for ofi, items in groups.items():
            items.sort(key=lambda item: int(item[0]["idx_in_output"]))
            indices = [int(item[0]["idx_in_output"]) for item in items]
            if indices != list(range(len(indices))):
                raise ValueError(
                    f"slot {ofi}: idx_in_output must be unique and contiguous 0..n-1, got {indices}"
                )
        plan.append((entry, groups))
    return plan


def _photon_dose(
    models: Models,
    ctvol,
    beam,
    cp,
    origin,
    spacing,
    shape,
    scale,
    voxel_mask=None,
    prepared_grid=None,
    prepared_selection=None,
    cone_cache=None,
):
    geo = BeamGeometry(
        gantry_deg=float(cp["gantry_angle"]),
        isocenter_mm=tuple((float(v) for v in beam["iso_center"])),
        sad_mm=float(beam.get("SAD", 1000.0)),
    )
    ap = _photon_aperture(beam, cp)
    region = (
        "full" if PHOTON_INFERENCE_REGION == "body_filled" else PHOTON_INFERENCE_REGION
    )
    dose = models.engine.predict_dose_grid(
        ctvol,
        geo,
        ap,
        origin,
        spacing,
        shape,
        region=region,
        amp=models.device.type == "cuda",
        voxel_mask=voxel_mask,
        prepared_grid=prepared_grid,
        prepared_selection=prepared_selection,
        cone_cache=cone_cache,
    )
    return dose * scale


def _photon_geometry_and_aperture(beam: dict, cp: dict) -> tuple:
    geometry = BeamGeometry(
        gantry_deg=float(cp["gantry_angle"]),
        isocenter_mm=tuple((float(v) for v in beam["iso_center"])),
        sad_mm=float(beam.get("SAD", 1000.0)),
    )
    return (geometry, _photon_aperture(beam, cp))


def _photon_query_cull_selection(
    models: Models,
    site: str,
    cutoff: float,
    beam: dict,
    cp: dict,
    prepared_grid,
    *,
    enabled: bool,
    minimum_cutoff: float,
    thorax_margin_mm: float,
):
    """Return the commissioned Photon thorax crop or the full-path marker."""
    if (
        not enabled
        or models.image_type not in ("ct", "mri")
        or site != "thx"
        or (float(cutoff) < float(minimum_cutoff))
        or (prepared_grid is None)
    ):
        return None
    geometry, aperture = _photon_geometry_and_aperture(beam, cp)
    return models.engine.aperture_margin_query_mask(
        prepared_grid, geometry, aperture, thorax_margin_mm
    )


def _g(v) -> str:
    """MetaImage-style number: integer if whole, else repr (parseable by ITK)."""
    v = float(v)
    return str(int(v)) if v == int(v) else repr(v)


def _mha_4d_header(ref_img, shape3d, n_frames: int) -> bytes:
    """Uncompressed MetaImage header matching ``sitk.JoinSeries``."""
    nz, ny, nx = (int(s) for s in shape3d)
    if n_frames <= 0:
        raise ValueError("a used output slot must contain at least one frame")
    if tuple((int(v) for v in ref_img.GetSize())) != (nx, ny, nz):
        raise ValueError(
            f"dose shape {shape3d} does not match source image size {ref_img.GetSize()}"
        )
    sx, sy, sz = ref_img.GetSpacing()
    ox, oy, oz = ref_img.GetOrigin()
    dr = ref_img.GetDirection()
    if len(dr) != 9:
        raise ValueError(
            f"expected a 3D source image, got direction of length {len(dr)}"
        )
    tm = [
        dr[0],
        dr[3],
        dr[6],
        0,
        dr[1],
        dr[4],
        dr[7],
        0,
        dr[2],
        dr[5],
        dr[8],
        0,
        0,
        0,
        0,
        1,
    ]
    return f"ObjectType = Image\nNDims = 4\nBinaryData = True\nBinaryDataByteOrderMSB = False\nCompressedData = False\nTransformMatrix = {' '.join((_g(v) for v in tm))}\nOffset = {_g(ox)} {_g(oy)} {_g(oz)} 0\nCenterOfRotation = 0 0 0 0\nElementSpacing = {_g(sx)} {_g(sy)} {_g(sz)} 1\nDimSize = {nx} {ny} {nz} {n_frames}\nAnatomicalOrientation = ????\nElementType = MET_FLOAT\nElementDataFile = LOCAL\n".encode(
        "ascii"
    )


class _StreamingSlotWriter:
    """One-pass, one-frame-memory writer for an uncompressed 4D MHA slot."""

    def __init__(self, out_path: Path, ref_img, shape3d, n_frames: int):
        self.out_path = Path(out_path)
        self.shape3d = tuple((int(v) for v in shape3d))
        self.n_frames = int(n_frames)
        self.next_index = 0
        self._finished = False
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self._out = open(self.out_path, "wb")
        self._out.write(_mha_4d_header(ref_img, self.shape3d, self.n_frames))

    def write(self, index: int, frame: np.ndarray) -> None:
        if self._finished:
            raise RuntimeError("cannot append to a finished output slot")
        if int(index) != self.next_index:
            raise ValueError(f"expected frame {self.next_index}, got {index}")
        frame = np.ascontiguousarray(frame, dtype=np.float32)
        if tuple(frame.shape) != self.shape3d:
            raise ValueError(
                f"frame {index}: shape {frame.shape}, expected {self.shape3d}"
            )
        self._out.write(memoryview(frame).cast("B"))
        self.next_index += 1

    def write_tensor(self, index: int, dose: torch.Tensor) -> None:
        """Synchronously transfer and append one device or host tensor."""
        frame = np.ascontiguousarray(dose.detach().cpu().numpy(), dtype=np.float32)
        self.write(index, frame)

    def finish(self) -> None:
        if self._finished:
            return
        if self.next_index != self.n_frames:
            raise ValueError(
                f"wrote {self.next_index} frames, expected {self.n_frames}"
            )
        self._out.close()
        self._finished = True

    def abort(self) -> None:
        if self._finished:
            return
        if not self._out.closed:
            self._out.close()
        self.out_path.unlink(missing_ok=True)
        self._finished = True


class _AsyncStreamingSlotWriter:
    """Single-frame D2H/write pipeline preserving exact frame order."""

    def __init__(self, out_path: Path, ref_img, shape3d, n_frames: int):
        self._writer = _StreamingSlotWriter(out_path, ref_img, shape3d, n_frames)
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="mha-writer"
        )
        self._pending: Future | None = None
        self._copy_stream: torch.cuda.Stream | None = None
        self._finished = False

    @staticmethod
    def _wait_and_write(
        writer: _StreamingSlotWriter,
        index: int,
        host: torch.Tensor,
        ready: torch.cuda.Event | None,
    ) -> None:
        if ready is not None:
            ready.synchronize()
        writer.write(index, host.numpy())

    def _join_pending(self) -> None:
        if self._pending is not None:
            pending, self._pending = (self._pending, None)
            pending.result()

    def write_tensor(self, index: int, dose: torch.Tensor) -> None:
        if self._finished:
            raise RuntimeError("cannot append to a finished output slot")
        self._join_pending()
        source = dose.detach()
        ready = None
        if source.device.type == "cuda":
            device = source.device
            if self._copy_stream is None:
                self._copy_stream = torch.cuda.Stream(device=device)
            elif self._copy_stream.device != device:
                raise ValueError(
                    "all frames in one output slot must use one CUDA device"
                )
            host = torch.empty(
                source.shape, dtype=torch.float32, device="cpu", pin_memory=True
            )
            producer = torch.cuda.current_stream(device)
            self._copy_stream.wait_stream(producer)
            with torch.cuda.stream(self._copy_stream):
                host.copy_(source, non_blocking=True)
                ready = torch.cuda.Event()
                ready.record(self._copy_stream)
            source.record_stream(self._copy_stream)
        else:
            host = source.to(device="cpu", dtype=torch.float32).contiguous()
        self._pending = self._executor.submit(
            self._wait_and_write, self._writer, int(index), host, ready
        )

    def finish(self) -> None:
        if self._finished:
            return
        self._join_pending()
        self._executor.shutdown(wait=True)
        self._writer.finish()
        self._finished = True

    def abort(self) -> None:
        if self._finished:
            return
        try:
            self._join_pending()
        except Exception:
            pass
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._writer.abort()
        self._finished = True


@torch.inference_mode()
def run(input_dir: Path, output_dir: Path, models: Models) -> None:
    """Process one invoke: read the mount, compute + route + write 10 output slots."""
    input_dir, output_dir = (Path(input_dir), Path(output_dir))
    meta_name = f"stacked-{models.modality}-beam-level-metadata.json"
    meta = json.loads((input_dir / meta_name).read_text())
    plan = _routing_plan(meta, models.modality)
    used_slots = {ofi for _, groups in plan for ofi in groups}
    active_writer: _StreamingSlotWriter | _AsyncStreamingSlotWriter | None = None
    cutoff_diagnostics = (
        _CutoffDiagnostics() if _env_flag("PRISMX_CUTOFF_DIAGNOSTICS") else None
    )
    n_dosemaps = 0
    legacy_mri_query_cull = _env_flag("PRISMX_PHOTON_MRI_QUERY_CULL", default=False)
    photon_query_cull = _env_flag(
        "PRISMX_PHOTON_QUERY_CULL", default=legacy_mri_query_cull
    )
    photon_query_cull_minimum_cutoff = float(
        os.environ.get("PRISMX_PHOTON_QUERY_CULL_MIN_CUTOFF", "3e-7")
    )
    photon_query_cull_thorax_margin = float(
        os.environ.get("PRISMX_PHOTON_QUERY_CULL_THORAX_MARGIN_MM", "80")
    )
    if (
        not math.isfinite(photon_query_cull_minimum_cutoff)
        or photon_query_cull_minimum_cutoff < 0.0
        or (not math.isfinite(photon_query_cull_thorax_margin))
        or (photon_query_cull_thorax_margin < 0.0)
    ):
        raise ValueError(
            "Photon query-cull cutoff and margin must be finite and non-negative"
        )
    print(
        f"[runtime-gates] query_cull={int(photon_query_cull)} query_cull_min_cutoff={photon_query_cull_minimum_cutoff:g} query_cull_thx_margin_mm={photon_query_cull_thorax_margin:g}",
        flush=True,
    )
    started = time.perf_counter()
    try:
        for entry, groups in plan:
            img_idx = int(entry["image_file_idx"])
            site = _REGION_TO_SITE[entry["anatomical_region"]]
            margin_value = os.environ.get(
                "PRISMX_ENCODER_MARGIN_THX"
                if site == "thx"
                else "PRISMX_ENCODER_MARGIN_ABD"
            )
            if margin_value:
                import prismx.model as photon_model

                photon_model.ENCODER_BOX_MARGIN_MM = float(margin_value)
                print(
                    f"[runtime-gates] encoder margin {site}={margin_value} mm",
                    flush=True,
                )
            scale = models.scale(site)
            slot_dir = (
                input_dir
                / "images"
                / f"radiation-dose-calculation-source-{models.image_type}-image-{img_idx + 1}"
            )
            arr, spacing, origin, ref_img = _read_image(_find_mha(slot_dir))
            shape = arr.shape
            ctvol = _ctvolume_for_image(models, arr, spacing, origin)
            photon_voxel_mask = None
            photon_prepared_grid = None
            photon_cone_cache = None
            if PHOTON_INFERENCE_REGION == "body_filled":
                photon_voxel_mask = _filled_body_voxel_mask(ctvol.density)
                photon_prepared_grid = models.engine.prepare_grid_query(
                    ctvol, origin, spacing, shape, photon_voxel_mask
                )
                if _env_flag("PRISMX_CONE_CACHE", default=True):
                    photon_cone_cache = {}
            for ofi, items in sorted(groups.items()):
                out_path = (
                    output_dir
                    / OUTPUT_IMAGE_ROOT
                    / f"stacked-radiation-dose-map-{ofi + 1}"
                    / OUTPUT_FILENAME
                )
                writer_type = (
                    _AsyncStreamingSlotWriter
                    if _env_flag("PRISMX_ASYNC_OUTPUT_WRITER", default=True)
                    else _StreamingSlotWriter
                )
                active_writer = writer_type(out_path, ref_img, shape, len(items))
                pos = 0
                while pos < len(items):
                    chunk = items[pos : pos + 1]
                    geos, states = ([None], {0: None})
                    for ci, item in enumerate(chunk):
                        oi = item[0]
                        _, beam, cp = item
                        prepared_selection = _photon_query_cull_selection(
                            models,
                            site,
                            float(oi["minimum_cutoff"]),
                            beam,
                            cp,
                            photon_prepared_grid,
                            enabled=photon_query_cull,
                            minimum_cutoff=photon_query_cull_minimum_cutoff,
                            thorax_margin_mm=photon_query_cull_thorax_margin,
                        )
                        dose = _photon_dose(
                            models,
                            ctvol,
                            beam,
                            cp,
                            origin,
                            spacing,
                            shape,
                            scale,
                            prepared_grid=photon_prepared_grid,
                            prepared_selection=prepared_selection,
                            cone_cache=photon_cone_cache,
                        )
                        cutoff = float(oi["minimum_cutoff"])
                        packed_stats = _apply_minimum_cutoff_(
                            dose, cutoff, cutoff_diagnostics
                        )
                        active_writer.write_tensor(int(oi["idx_in_output"]), dose)
                        if cutoff_diagnostics is not None:
                            cutoff_diagnostics.add(cutoff, packed_stats)
                        del dose
                        n_dosemaps += 1
                    pos += len(chunk)
                active_writer.finish()
                active_writer = None
            del ctvol, arr
        _write_placeholders(output_dir, used_slots)
    finally:
        if active_writer is not None:
            active_writer.abort()
    elapsed = time.perf_counter() - started
    if cutoff_diagnostics is not None:
        cutoff_diagnostics.report()
    per_map = elapsed / n_dosemaps if n_dosemaps else 0.0
    print(
        f"[inference] wrote {n_dosemaps} dosemaps in {elapsed:.3f}s ({per_map:.3f}s/dosemap)",
        flush=True,
    )


def _write_placeholders(output_dir: Path, used_slots: set[int]) -> None:
    """Complete the ten-slot contract with uncompressed zero placeholders."""
    import SimpleITK as sitk

    for s in range(N_OUTPUT_SLOTS):
        d = output_dir / OUTPUT_IMAGE_ROOT / f"stacked-radiation-dose-map-{s + 1}"
        d.mkdir(parents=True, exist_ok=True)
        if s not in used_slots:
            ph = sitk.Image(1, 1, sitk.sitkFloat32)
            sitk.WriteImage(ph, str(d / OUTPUT_FILENAME), useCompression=False)


def _default_cfg(modality: str, image_type: str) -> dict:
    return {
        "engine": str(_REPO / "weights/photon.pt"),
        "pdd": str(_REPO / "commissioning/synthrad_photon_pdd.npz"),
        "net": str(_REPO / "weights/mr-density.pt"),
    }


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="DoseRAD2026 inference (local test harness)"
    )
    ap.add_argument("--modality", required=True, choices=("photon",))
    ap.add_argument("--image-type", required=True, choices=("ct", "mri"))
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    models = Models(
        a.modality, a.image_type, a.device, _default_cfg(a.modality, a.image_type)
    )
    run(a.input, a.output, models)
    print("INFERENCE-DONE", flush=True)


if __name__ == "__main__":
    main()
