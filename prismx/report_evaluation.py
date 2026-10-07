"""Evaluate released output frames without retaining multi-gigabyte 4-D stacks.

Run from the release root with ``python -m prismx.report_evaluation``. The
collector replaces serialization only. The released prediction, density,
geometry, support and cutoff paths are unchanged. GT is loaded in the output
collector, after prediction, and never defines inference support.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def array_sha(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array, dtype="<f4")).cast("B")).hexdigest()


def configure(modality, image_type):
    """Same environment and file checks as the release CLI, before imports."""
    manifest = json.loads((ROOT / "release-manifest.json").read_text())
    env = json.loads((ROOT / "configs" / f"{modality}-{image_type}.json").read_text())
    for name in list(os.environ):
        if name.startswith("PRISMX_"):
            del os.environ[name]
    os.environ.update(env)
    os.environ.update({"PRISMX_MODALITY": modality, "PRISMX_IMAGE_TYPE": image_type,
                      "PRISMX_DEVICE": "cuda", "PRISMX_SKIP_WEIGHT_VERIFY": "0",
                      "PRISMX_GIT_REV": "v1.0.0",
                      "PRISMX_EXPECT_ENGINE_SHA256": manifest["files"][f"weights/{modality}.pt"],
                      "PRISMX_EXPECT_MR_NET_SHA256": manifest["files"]["weights/mr-density.pt"] if image_type == "mri" else "none",
                      "PRISMX_EXPECT_COMMISSIONING_SHA256": manifest["commissioning_sha256"]})
    checked = {}
    for relative, expected in manifest["files"].items():
        if relative.startswith("weights/") and relative.endswith(".pt"):
            if relative not in (f"weights/{modality}.pt", "weights/mr-density.pt" if image_type == "mri" else ""):
                continue
        observed = sha(ROOT / relative)
        if observed != expected:
            raise ValueError(f"Release hash mismatch: {relative}")
        checked[relative] = observed
    return checked


def prepare_case(data, modality, image_type, case_id, out, cutoff, limit=0):
    source = Path(data) / modality / "training" / case_id
    plan_path = source / f"{case_id}.json"
    plan = json.loads(plan_path.read_text())
    metadata = copy.deepcopy(plan)
    metadata.update(image_file_idx=0, anatomical_region="abdominal" if "ABB" in case_id else "thoracic")
    records, new_beams = [], []
    for beam in metadata["beams"]:
        b = int(beam["beam_idx"])
        if modality == "photon":
            cps = []
            for cp in beam["control_points"]:
                if limit and len(records) >= limit:
                    break
                c = int(cp["cp_idx"])
                gt = source / "dose" / f"Dose_B{b}_CP{c:03d}.mha"
                if not gt.is_file():
                    raise FileNotFoundError(gt)
                cp["output_info"] = dict(output_file_idx=0, idx_in_output=len(records), minimum_cutoff=cutoff)
                records.append(dict(gt=str(gt), beam=b, cp=c))
                cps.append(cp)
            beam["control_points"] = cps
            if cps:
                new_beams.append(beam)
        else:
            rays = []
            for ray in beam["rays"]:
                r = int(ray["ray_idx"])
                layers = []
                for layer in ray["beamlets"]:
                    if limit and len(records) >= limit:
                        break
                    l = int(layer["beamlet_idx"])
                    gt = source / "dose" / f"Dose_B{b}_R{r}_L{l}.mha"
                    if not gt.is_file():
                        raise FileNotFoundError(gt)
                    layer["output_info"] = dict(output_file_idx=0, idx_in_output=len(records), minimum_cutoff=cutoff)
                    records.append(dict(gt=str(gt), beam=b, ray=r, layer=l,
                                        gantry=beam["gantry_angle"], ray_source=ray["ray_source"],
                                        ray_target=ray["ray_target"], iso_center=plan["iso_center"]))
                    layers.append(layer)
                ray["beamlets"] = layers
                if layers:
                    rays.append(ray)
            beam["rays"] = rays
            if rays:
                new_beams.append(beam)
    metadata["beams"] = new_beams
    if not records:
        raise ValueError("No segments")
    inputs = out / "input"
    inputs.mkdir()
    slot = inputs / "images" / f"radiation-dose-calculation-source-{image_type}-image-1"
    slot.mkdir(parents=True)
    image = source / "image" / ("mr.mha" if image_type == "mri" else "ct.mha")
    # All symlinks and metadata are new outputs; source cases are immutable.
    (slot / "input.mha").symlink_to(image)
    save(inputs / f"stacked-{modality}-beam-level-metadata.json", [metadata])
    save(out / "segments.json", records)
    return inputs, records, {"plan_sha256": sha(plan_path), "image_sha256": sha(image)}


def binned_profile(wed, dose, start):
    edges = np.arange(start, wed.max() + 1.0, 1.0)
    return edges, np.asarray([dose[(wed >= d) & (wed < d + 1)].mean()
                            if np.any((wed >= d) & (wed < d + 1)) else 0.0
                            for d in edges[:-1]])


def r80(wed, dose):
    if len(wed) < 10:
        return None
    edges, profile = binned_profile(wed, dose, wed.min())
    if len(profile) < 3 or profile.max() <= 0:
        return None
    peak = int(profile.argmax())
    below = np.flatnonzero(profile[peak:] < 0.8 * profile.max())
    return float(edges[peak + below[0]] + 0.5) if len(below) else None


def range_score(frame, ref, record, ct, spacing, origin):
    import torch
    from prismx.proton_data import beamlet_geometry_from_ray
    from prismx.proton_grid import ProtonGridSpec, build_proton_grid
    # Work only on CPU and only after prediction. This mask is a scoring gate.
    indices = np.argwhere(ref > .01 * float(ref.max()))
    points = torch.from_numpy(np.ascontiguousarray(indices[:, ::-1] * np.asarray(spacing) + np.asarray(origin), dtype=np.float32))
    geometry = beamlet_geometry_from_ray(record["gantry"], record["iso_center"], record["ray_source"], record["ray_target"])
    uvd = geometry.world_to_beamlet(points)
    on_axis = torch.linalg.vector_norm(uvd[:, :2], dim=1) < 3.0
    uvd, indices = uvd[on_axis], indices[on_axis.numpy()]
    if len(indices) < 10:
        return {"valid": False, "reason": "sparse_core"}
    grid = build_proton_grid(ct, geometry.beam, geometry.spot_uv, ProtonGridSpec())
    inside = grid.contains(uvd)
    uvd, indices = uvd[inside], indices[inside.numpy()]
    if len(indices) < 10:
        return {"valid": False, "reason": "sparse_in_grid_core"}
    wed = grid.sample(grid.wed, uvd).numpy()
    gt = ref[tuple(indices.T)]
    pred = frame[tuple(indices.T)]
    g, p = r80(wed, gt), r80(wed, pred)
    _, profile = binned_profile(wed, gt, np.floor(wed.min()))
    complete = len(profile) > 3 and profile.max() > 0 and bool(np.any(profile[int(profile.argmax()):] < .5 * profile.max()))
    valid = p is not None and g is not None and complete
    return {"valid": valid, "reference_r80_mm": g, "predicted_r80_mm": p,
            "complete_reference_falloff": complete, "delta_r80_mm": p - g if valid else None}


def frame_metrics(frame, reference):
    if frame.shape != reference.shape or not np.isfinite(frame).all() or np.any(frame < 0):
        raise ValueError("Invalid output dose frame")
    peak = float(reference.max())
    if peak <= 0:
        raise ValueError("Empty reference dose")
    mask = reference >= .1 * peak
    return float(np.mean(np.abs(frame[mask].astype(np.float64) - reference[mask])) / peak)


def build(args):
    import torch
    protocol = json.loads(args.protocol.read_text())
    if args.case not in protocol["cases"] or args.task not in protocol["tasks"]:
        raise ValueError("Case/task outside frozen protocol")
    modality, image_type = args.task.split("_")
    checked = configure(modality, image_type)
    if args.preflight:
        print(json.dumps({"status": "PREFLIGHT_PASS", "task": args.task, "verified_files": len(checked), "torch": torch.__version__}), flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for the frozen release settings")
    # Cap this process's allocator while preserving other workloads.
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, args.allocator_cap_gib * 1024**3 / total), 0)
    args.out.mkdir(parents=True, exist_ok=False)
    inputs, records, source_hashes = prepare_case(args.data, modality, image_type, args.case, args.out,
                                               protocol["minimum_cutoff_gy"], args.limit)
    from prismx import inference as io
    from prismx import runtime
    torch.set_num_threads(4)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(2)
    state = {"rows": [], "prediction_seconds": [], "density_seconds": [], "offset": 0}
    original_density = io._ctvolume_for_image
    original_writer = io._StreamingSlotWriter
    selected = set(np.linspace(0, len(records) - 1, min(150, len(records)), dtype=int))

    def density(*a, **kw):
        if "ct_gpu" in state:
            return state["ct_gpu"]
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = original_density(*a, **kw)
        torch.cuda.synchronize()
        state["density_seconds"].append(time.perf_counter() - started)
        state["ct_gpu"] = result
        if modality == "proton":
            state["ct"] = result.to("cpu")
        return result

    def timed(function):
        def wrapped(*a, **kw):
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = function(*a, **kw)
            torch.cuda.synchronize()
            state["prediction_seconds"].append(time.perf_counter() - started)
            return result
        return wrapped

    class Collector:
        def __init__(self, path, image, shape, n_frames):
            self.image, self.next_index, self.n_frames = image, 0, n_frames
            if "pred" not in state:
                state["pred"] = np.zeros(shape, np.float64)
                state["ref"] = np.zeros(shape, np.float64)
            elif tuple(shape) != state["pred"].shape:
                raise ValueError("Batch geometry changed")
            state["spacing"] = tuple(image.GetSpacing())
            state["origin"] = tuple(image.GetOrigin())

        def write(self, index, frame):
            if index != self.next_index:
                raise ValueError("Frame order mismatch")
            global_index = state["offset"] + index
            rec = records[global_index]
            gt_image = sitk.ReadImage(rec["gt"])
            for attr in ("GetSize", "GetSpacing", "GetOrigin", "GetDirection"):
                if not np.allclose(getattr(gt_image, attr)(), getattr(self.image, attr)(), rtol=0, atol=1e-5):
                    raise ValueError(f"GT/input geometry mismatch: {attr}")
            gt = sitk.GetArrayFromImage(gt_image).astype(np.float32)
            cutoff = np.float32(protocol["minimum_cutoff_gy"])
            if np.any((frame > 0) & (frame <= cutoff)):
                raise ValueError("Output cutoff contract failed")
            row = {"segment": int(global_index), "beam_mae": frame_metrics(frame, gt),
                   "gt_file_sha256": sha(rec["gt"]), "prediction_array_sha256": array_sha(frame)}
            if modality == "proton" and global_index in selected:
                row["range"] = range_score(frame, gt, rec, state["ct"], state["spacing"], state["origin"])
            state["pred"] += frame
            state["ref"] += gt
            if global_index == 0:
                # Check capture against the actual released serializer, not
                # a second reconstruction of the frame.
                path = args.out / "first-frame.mha"
                writer = original_writer(path, self.image, frame.shape, 1)
                writer.write(0, frame)
                writer.finish()
                stored = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
                if not np.array_equal(stored.reshape(frame.shape), frame):
                    raise ValueError("Serialized and scored frame differ")
                row["serialization_bit_equal"] = True
            state["rows"].append(row)
            self.next_index += 1
            if self.next_index % 10 == 0 or self.next_index == self.n_frames:
                count = len(state["rows"])
                save(args.out / "progress.json", {"status": "BUILDING", "frames": count, "total": len(records)})
                print(f"PROGRESS {args.task} {args.case} {count}/{len(records)}", flush=True)

        def write_tensor(self, index, dose):
            self.write(index, np.ascontiguousarray(dose.detach().cpu().numpy(), dtype=np.float32))

        def finish(self):
            if self.next_index != self.n_frames:
                raise ValueError("Missing output frames")

        def abort(self):
            pass

    io._StreamingSlotWriter = Collector
    io._ctvolume_for_image = density
    if modality == "photon":
        io._photon_dose = timed(io._photon_dose)
    else:
        from prismx import proton_inference
        proton_inference.beamlet_dose = timed(proton_inference.beamlet_dose)
    start = time.perf_counter()
    models = runtime.Models(modality, image_type, "cuda", runtime._default_cfg(modality, image_type))
    metadata_path = inputs / f"stacked-{modality}-beam-level-metadata.json"
    full_meta = json.loads(metadata_path.read_text())
    batch_size = int(protocol["inference_batch_segments"])
    for offset in range(0, len(records), batch_size):
        state["offset"] = offset
        stop = min(offset + batch_size, len(records))
        batch = copy.deepcopy(full_meta)
        beams = []
        for beam in batch[0]["beams"]:
            if modality == "photon":
                beam["control_points"] = [cp for cp in beam["control_points"]
                                          if offset <= cp["output_info"]["idx_in_output"] < stop]
                items = beam["control_points"]
            else:
                rays = []
                for ray in beam["rays"]:
                    ray["beamlets"] = [b for b in ray["beamlets"] if offset <= b["output_info"]["idx_in_output"] < stop]
                    if ray["beamlets"]:
                        rays.append(ray)
                beam["rays"] = rays
                items = [b for ray in rays for b in ray["beamlets"]]
            for item in items:
                item["output_info"]["idx_in_output"] -= offset
            if items:
                beams.append(beam)
        batch[0]["beams"] = beams
        batch_input = args.out / f"input-batch-{offset:04d}"
        batch_input.mkdir()
        (batch_input / "images").symlink_to(inputs / "images", target_is_directory=True)
        save(batch_input / metadata_path.name, batch)
        runtime.run(batch_input, args.out / "output", models)
    state["offset"] = 0
    if len(state["rows"]) != len(records):
        raise ValueError("Not all planned segments scored")
    np.savez_compressed(args.out / "plan.npz", pred=state["pred"], ref=state["ref"],
                        spacing=np.asarray(state["spacing"]), origin=np.asarray(state["origin"]))
    result = {"status": "BUILT", "task": args.task, "case": args.case, "segments": len(records),
              "complete_plan": args.limit == 0, "source_hashes": source_hashes,
              "protocol_sha256": sha(args.protocol), "evaluator_sha256": sha(__file__),
              "verified_release_files": checked, "torch": torch.__version__,
              "gpu": torch.cuda.get_device_name(), "wall_seconds_including_metrics": time.perf_counter() - start,
              "shared_gpu_authorized": True, "allocator_cap_gib": args.allocator_cap_gib,
              "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
              "peak_cuda_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
              "timings_are_exploratory_shared_gpu": True,
              "prediction_seconds": state["prediction_seconds"], "density_seconds": state["density_seconds"],
              "frames": state["rows"], "plan_sha256": sha(args.out / "plan.npz")}
    save(args.out / "build.json", result)
    save(args.out / "progress.json", {"status": "BUILT", "frames": len(records), "total": len(records)})
    print("BUILD_COMPLETE", args.task, args.case, flush=True)


def score(args):
    import pymedphys
    protocol = json.loads(args.protocol.read_text())
    build_record = json.loads((args.out / "build.json").read_text())
    if sha(args.protocol) != build_record["protocol_sha256"] or sha(args.out / "plan.npz") != build_record["plan_sha256"]:
        raise ValueError("Protocol or output changed")
    z = np.load(args.out / "plan.npz", allow_pickle=False)
    pred, ref, spacing = z["pred"], z["ref"], z["spacing"]
    peak = float(ref.max())
    masks = {"high": ref >= .8 * peak, "mid": (ref >= .3 * peak) & (ref < .8 * peak),
             "low": (ref >= .1 * peak) & (ref < .3 * peak)}
    mae = {k: float(np.abs(pred[m] - ref[m]).mean() / peak) if m.any() else None for k, m in masks.items()}
    axes = tuple(np.arange(n, dtype=np.float64) * s for n, s in zip(ref.shape, spacing[::-1]))
    g = protocol["gamma"]
    gamma = pymedphys.gamma(axes, ref, axes, pred, dose_percent_threshold=g["dose_percent"],
                           distance_mm_threshold=g["distance_mm"], lower_percent_dose_cutoff=g["lower_percent_dose_cutoff"],
                           interp_fraction=g["interp_fraction"], max_gamma=g["max_gamma"],
                           local_gamma=g["local"], ram_available=4 * 10**9)
    finite = gamma[~np.isnan(gamma)]
    if not finite.size:
        raise ValueError("No gamma-evaluable voxels")
    ranges = [r["range"]["delta_r80_mm"] for r in build_record["frames"] if r.get("range", {}).get("valid")]
    result = {"task": build_record["task"], "case": build_record["case"],
              "complete_plan": build_record["complete_plan"], "segments": build_record["segments"],
              "mae_bands": mae, "gamma_fraction": float(np.mean(finite <= 1)), "gamma_voxels": int(finite.size),
              "integral_ratio": float(pred.sum() / ref.sum()),
              "beam_mae": float(np.mean([r["beam_mae"] for r in build_record["frames"]])),
              "r80_valid": len(ranges), "r80_attempted": sum("range" in r for r in build_record["frames"]),
              "r80_mae_mm": float(np.mean(np.abs(ranges))) if ranges else None,
              "r80_bias_mm": float(np.mean(ranges)) if ranges else None,
              "prediction_ms_per_segment": 1000 * float(np.mean(build_record["prediction_seconds"])),
              "density_seconds": build_record["density_seconds"], "protocol_sha256": sha(args.protocol),
              "build_sha256": sha(args.out / "build.json"), "pymedphys": pymedphys.__version__}
    save(args.out / "metrics.json", result)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("build", "score"), required=True)
    p.add_argument("--protocol", type=Path, default=ROOT / "experiments/report/release-evaluation.json")
    p.add_argument("--data", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--task", choices=("photon_ct", "photon_mri", "proton_ct", "proton_mri"))
    p.add_argument("--case", choices=("1ABB118", "1THB011"))
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="Smoke test only; never a complete-plan result")
    p.add_argument("--allocator-cap-gib", type=float, default=28)
    a = p.parse_args()
    if a.mode == "build":
        if a.task is None or a.case is None or (not a.preflight and a.data is None):
            p.error("build requires task, case and data")
        build(a)
    else:
        score(a)


if __name__ == "__main__":
    main()
