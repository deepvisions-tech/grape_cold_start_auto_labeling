"""Regenerate paper tables and statistical intervals from anonymous observations.

Reference tables are read only by the final comparison, never by calculations.
Detector AP is an archived COCOeval input. Presence aggregation, summaries,
paired tests, label-recovery statistics, annotation summaries and the AdaBN
setting-selection interval are recomputed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
from itertools import product
from pathlib import Path
import re

import numpy as np
from scipy.stats import t as student_t


METHODS = {
    "SAM3-only": "sam3_only",
    "SAM3 + image exemplar": "sam3_exemplar",
    "SAM3 + image exemplar + DINOv2": "sam3_exemplar_dinov2",
    "YOLO26x relabeler": "yolo26x_relabeler",
}
WORKFLOWS = {
    "Fully manual": "manual",
    "SAM3 + image exemplar + DINOv2 then human review": "sam3_exemplar_dinov2",
    "YOLO26x then human review": "yolo26x",
}
SCENARIOS = {"cold-start synthetic-only": "cold", "real-only": "real",
             "synthetic-pretrain + fine-tune": "finetune"}


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields=None):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fields or list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def exact_sign_flip(values):
    x = np.asarray(values, dtype=float)
    if not 1 <= len(x) <= 20:
        raise ValueError("Exact sign-flip requires 1 to 20 independent pairs")
    bits = (np.arange(1 << len(x), dtype=np.uint32)[:, None] >> np.arange(len(x), dtype=np.uint32)) & 1
    stats = np.abs((1.0 - 2.0 * bits) @ x / len(x))
    return float(np.mean(stats >= abs(x.mean()) - 16 * np.finfo(float).eps))


def holm(pvalues):
    order = sorted(range(len(pvalues)), key=pvalues.__getitem__)
    adjusted, previous = [0.0] * len(pvalues), 0.0
    for rank, i in enumerate(order):
        previous = max(previous, min(1.0, (len(order) - rank) * pvalues[i]))
        adjusted[i] = previous
    return adjusted


def detector_records(bundle, protocol):
    records = read_csv(bundle / "detector_runs.csv")
    if len(records) != protocol["detector_runs"]:
        raise ValueError("Detector run count differs")
    by_id = {r["run_id"]: r for r in records}
    if len(by_id) != len(records):
        raise ValueError("Duplicate detector run ID")
    grouped = defaultdict(list)
    for r in records:
        r["train_seed"] = int(r["train_seed"])
        for key in ("object_tp", "object_fp", "object_fn", "precision", "recall", "f1",
                    "image_tp", "image_tn", "image_fp", "image_fn", "ap50", "ap50_95"):
            r[key] = float(r[key])
            if not np.isfinite(r[key]) or r[key] < 0:
                raise ValueError(f"Invalid detector input: {r['run_id']} {key}")
        tp, fp, fn = (r["object_" + k] for k in ("tp", "fp", "fn"))
        metrics = (tp / (tp + fp) if tp + fp else 0, tp / (tp + fn) if tp + fn else 0,
                   2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0)
        if not np.allclose(metrics, [r[k] for k in ("precision", "recall", "f1")], atol=1e-12, rtol=0):
            raise ValueError("Operating-point metrics differ from TP/FP/FN")
        positive, background = r["image_tp"] + r["image_fn"], r["image_tn"] + r["image_fp"]
        r.update(positive_tiles=positive, background_tiles=background,
                 false_alert_image_rate=r["image_fp"] / background,
                 false_alert_images=r["image_fp"], tile_level_recall=r["image_tp"] / positive,
                 image_level_tp=r["image_tp"], false_alert_tiles=r["image_fp"],
                 false_alert_tile_fraction=r["image_fp"] / background)
        grouped[r["arm"], r["real_positive_budget"], r["period"], r["composition"]].append(r)
    for key, rows in grouped.items():
        rows.sort(key=lambda r: r["train_seed"])
        if [r["train_seed"] for r in rows] != protocol["seed_order"]:
            raise ValueError(f"Unpaired or duplicate seeds: {key}")

    tiles = read_csv(bundle / "test_tiles.csv")
    tile_index = {r["tile_id"]: r for r in tiles}
    if len(tile_index) != len(tiles):
        raise ValueError("Duplicate tile ID")
    frame_tiles, positive_frames, frames = Counter(), defaultdict(set), defaultdict(set)
    for row in tiles:
        frame_tiles[row["frame_id"]] += 1
        frames[row["period"]].add(row["frame_id"])
        if int(row["gt_boxes"]):
            positive_frames[row["period"]].add(row["frame_id"])
    if set(frame_tiles.values()) != {8}:
        raise ValueError("Every source frame must contain exactly eight tiles")
    seen = set()
    with gzip.open(bundle / "deployment_presence.jsonl.gz", "rt") as f:
        for line in f:
            data = json.loads(line)
            r = by_id[data["run_id"]]
            if r["composition"] != "deployment" or r["run_id"] in seen:
                raise ValueError("Unexpected or duplicate deployment presence record")
            seen.add(r["run_id"])
            ids = data["positive_tile_ids"]
            if len(ids) != len(set(ids)) or any(i not in tile_index or tile_index[i]["period"] != r["period"] for i in ids):
                raise ValueError("Invalid deployment tile membership")
            ntp = sum(int(tile_index[i]["gt_boxes"]) > 0 for i in ids)
            if ntp != r["image_tp"] or len(ids) - ntp != r["image_fp"]:
                raise ValueError("Sparse presence and operating-point counts disagree")
            pred = {tile_index[i]["frame_id"] for i in ids}
            pos = positive_frames[r["period"]]
            bg = frames[r["period"]] - pos
            r.update(positive_frames=len(pos), background_frames=len(bg),
                     source_frame_tp=len(pred & pos), source_frame_recall=len(pred & pos) / len(pos),
                     false_alert_frames=len(pred & bg), false_alert_frame_fraction=len(pred & bg) / len(bg))
    if seen != {r["run_id"] for r in records if r["composition"] == "deployment"}:
        raise ValueError("Deployment presence coverage differs")
    return grouped, records


def rows_for(grouped, arm, budget, period, composition="curated"):
    return grouped[arm, str(budget) if budget not in (None, "") else "", period, composition]


def generate_detector_tables(layouts, grouped, protocol):
    outputs = {}
    indices = np.random.default_rng(protocol["detector_bootstrap_seed"]).integers(
        0, len(protocol["seed_order"]), size=(protocol["detector_bootstrap_resamples"], len(protocol["seed_order"])))
    for name, layout in layouts.items():
        if name.startswith("low_budget_subset_"):
            continue  # These require crossed-factor resampling, not 15 independent seeds.
        if "paired_effects" in name:
            rows, pvalues = [], []
            for meta in layout["rows"]:
                row = dict(meta)
                period, budget = row["test_period"], row.get("real_positive_budget")
                left, right = "finetune", "real"
                if name.startswith("generation"):
                    left, right = "no_adabn", "prompt"
                elif name.startswith("synthetic_only"):
                    left, right = "cold", "no_adabn"
                elif name.startswith("cold_start"):
                    left, right = "finetune", "cold"
                    budget = 10
                composition = row["composition"]
                a = rows_for(grouped, left, budget, period, composition)
                b = rows_for(grouped, right, None if name.startswith("cold_start") else budget, period, composition)
                diff = np.asarray([r["ap50_95"] for r in a]) - np.asarray([r["ap50_95"] for r in b])
                lo, hi = np.quantile(diff[indices].mean(axis=1), [.025, .975], method="linear")
                raw_p = exact_sign_flip(diff)
                row.update(n_seed_pairs=str(len(diff)), ap50_95_paired_effect=f"{diff.mean():.8f}",
                           bootstrap_ci_lower=f"{lo:.8f}", bootstrap_ci_upper=f"{hi:.8f}",
                           n_positive=str(np.count_nonzero(diff > 0)), raw_permutation_p=f"{raw_p:.8f}")
                pvalues.append(raw_p)
                rows.append(row)
            if "holm_adjusted_p" in layout["fields"]:
                for row, value in zip(rows, holm(pvalues)):
                    row["holm_adjusted_p"] = f"{value:.8f}"
            outputs[name] = rows
        elif any(k.endswith("_mean") for k in layout["fields"]):
            rows = []
            for meta in layout["rows"]:
                row = dict(meta)
                if name.startswith("generation"):
                    arm = "prompt" if row["generation_arm"] == "prompt-only" else "no_adabn"
                elif name.startswith("synthetic_only"):
                    arm = "cold" if row["arm"] == "+AdaBN" else "no_adabn"
                else:
                    arm = SCENARIOS[row["scenario"]]
                data = rows_for(grouped, arm, row.get("real_positive_budget"), row["test_period"], row["composition"])
                row["n_seeds"] = str(len(data))
                for k in ("positive_frames", "background_frames", "positive_tiles", "background_tiles"):
                    if k in row:
                        values = {r[k] for r in data}
                        if len(values) != 1:
                            raise ValueError("Evaluation population differs across seeds")
                        row[k] = str(int(values.pop()))
                for key in layout["fields"]:
                    if key.endswith("_mean"):
                        metric = key[:-5]
                        x = np.asarray([r[metric] for r in data])
                        for suffix, value in (("mean", x.mean()), ("sample_sd", x.std(ddof=1)), ("gap", np.ptp(x))):
                            row[metric + "_" + suffix] = f"{value:.6f}"
                rows.append(row)
            outputs[name] = rows
    return outputs


def subset_records(bundle, protocol):
    """Validate the complete paired subset-by-training-seed factorial design."""
    settings = protocol["subset_resampling"]
    expected = set(product(
        ("real", "finetune"), settings["budgets"], settings["periods"],
        settings["data_subset_seed_order"], settings["training_seed_order"],
    ))
    records = read_csv(bundle / "subset_resampling_runs.csv")
    if len(records) != settings["runs"] or settings["runs"] != len(expected):
        raise ValueError("Subset-resampling run count differs from the factorial design")
    cells, run_ids = {}, set()
    for raw in records:
        row = dict(raw)
        if not row["run_id"] or row["run_id"] in run_ids:
            raise ValueError("Duplicate or empty subset-resampling run ID")
        run_ids.add(row["run_id"])
        for key in ("real_positive_budget", "data_subset_seed", "train_seed"):
            row[key] = int(row[key])
        cell = (row["arm"], row["real_positive_budget"], row["period"],
                row["data_subset_seed"], row["train_seed"])
        if cell in cells or cell not in expected or row["composition"] != "curated":
            raise ValueError(f"Unexpected or duplicate subset-resampling cell: {cell}")
        for key in ("object_tp", "object_fp", "object_fn"):
            row[key] = float(row[key])
            if not np.isfinite(row[key]) or row[key] < 0 or not row[key].is_integer():
                raise ValueError(f"Invalid subset-resampling count: {key}")
        for key in ("precision", "recall", "f1", "ap50", "ap50_95", "applied_conf"):
            row[key] = float(row[key])
            if not np.isfinite(row[key]) or not 0 <= row[key] <= 1:
                raise ValueError(f"Invalid subset-resampling metric: {key}")
        tp, fp, fn = (row["object_" + key] for key in ("tp", "fp", "fn"))
        metrics = (tp / (tp + fp) if tp + fp else 0, tp / (tp + fn) if tp + fn else 0,
                   2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0)
        if not np.allclose(metrics, [row[k] for k in ("precision", "recall", "f1")], atol=1e-12, rtol=0):
            raise ValueError("Subset operating-point metrics differ from TP/FP/FN")
        cells[cell] = row
    if set(cells) != expected:
        raise ValueError("Incomplete subset-resampling pairing")
    return cells


def generate_subset_tables(layouts, cells, protocol):
    settings = protocol["subset_resampling"]
    if settings["permutation_unit"] != "data_subset_seed_mean":
        raise ValueError("Subset-resampling exact tests require subset-mean permutation units")
    subsets, seeds = settings["data_subset_seed_order"], settings["training_seed_order"]
    ns, nt = len(subsets), len(seeds)
    rng = np.random.default_rng(settings["bootstrap_seed"])
    # Independent factor draws; share both matrices across the six comparisons.
    subset_indices = rng.integers(0, ns, size=(settings["bootstrap_resamples"], ns))
    train_indices = rng.integers(0, nt, size=(settings["bootstrap_resamples"], nt))

    def matrix(arm, budget, period, metric):
        return np.asarray([[cells[arm, budget, period, subset, seed][metric]
                            for seed in seeds] for subset in subsets])

    aggregate_name = "low_budget_subset_results.csv"
    paired_name = "low_budget_subset_paired_effects.csv"
    if len(layouts[paired_name]["rows"]) != settings["holm_family_size"]:
        raise ValueError("Subset-resampling Holm family size differs")
    aggregates, paired, pvalues = [], [], []
    for meta in layouts[aggregate_name]["rows"]:
        row = dict(meta)
        row.update(n_data_subset_seeds=str(ns), n_training_seeds_per_subset=str(nt), n_runs=str(ns * nt))
        for metric in ("precision", "recall", "f1", "ap50", "ap50_95"):
            x = matrix(SCENARIOS[row["scenario"]], int(row["real_positive_budget"]), row["test_period"], metric)
            for suffix, value in (("mean", x.mean()), ("sample_sd", x.std(ddof=1)), ("gap", np.ptp(x))):
                row[metric + "_" + suffix] = f"{value:.6f}"
        aggregates.append(row)
    for meta in layouts[paired_name]["rows"]:
        row = dict(meta)
        budget, period = int(row["real_positive_budget"]), row["test_period"]
        diff = matrix("finetune", budget, period, "ap50_95") - matrix("real", budget, period, "ap50_95")
        sampled = np.take_along_axis(diff[subset_indices], train_indices[:, None, :], axis=2)
        lo, hi = np.quantile(sampled.mean(axis=(1, 2)), [.025, .975], method="linear")
        # The independent permutation units are the five subset means.
        subset_means = diff.mean(axis=1)
        pvalue = exact_sign_flip(subset_means)
        row.update(n_data_subset_seeds=str(ns), n_training_seeds=str(nt), n_paired_cells=str(ns * nt),
                   ap50_95_paired_effect=f"{diff.mean():.8f}", two_way_bootstrap_ci_lower=f"{lo:.8f}",
                   two_way_bootstrap_ci_upper=f"{hi:.8f}", n_positive_cells=str(np.count_nonzero(diff > 0)),
                   n_positive_subset_means=str(np.count_nonzero(subset_means > 0)),
                   permutation_unit="data_subset_seed_mean", raw_permutation_p=f"{pvalue:.8f}")
        pvalues.append(pvalue)
        paired.append(row)
    for row, adjusted in zip(paired, holm(pvalues)):
        row["holm_adjusted_p"] = f"{adjusted:.8f}"
    return {aggregate_name: aggregates, paired_name: paired}


def label_results(bundle, layout, protocol):
    from grape_auto_labeling.reproduction.statistics import analyze_records
    rows = read_csv(bundle / "label_recovery_images.csv")
    order = {}
    for row in rows:
        rank = int(row["analysis_order"])
        if rank in order and order[rank] != row["image_id"]:
            raise ValueError("Analysis order assigns multiple images to one rank")
        order[rank] = row["image_id"]
    if sorted(order) != list(range(1,len(order)+1)):
        raise ValueError("Incomplete label-recovery analysis order")
    result = analyze_records(rows,
        comparisons=[("sam3_exemplar_dinov2", "sam3_exemplar")],
        gate_comparison=("sam3_exemplar_dinov2", "sam3_exemplar_dinov2_pre_gate"),
        bootstrap_iterations=protocol["label_bootstrap_resamples"],
        permutation_iterations=protocol["label_permutations"], seed=protocol["label_seed"],
        image_order=[order[k] for k in sorted(order)], permutation_seed=protocol["label_permutation_seed"])
    output = []
    for meta in layout["rows"]:
        row = dict(meta)
        key = METHODS[row["method"]]
        source = result["methods"][key]
        row.update(uses_cross_image_exemplar=str(key in {"sam3_exemplar", "sam3_exemplar_dinov2"}).lower(),
                   uses_dinov2=str(key == "sam3_exemplar_dinov2").lower(),
                   trained_method=str(key == "yolo26x_relabeler").lower(),
                   n_images=str(source["n_images"]), n_gt_boxes=str(source["totals"]["tp"] + source["totals"]["fn"]))
        row.update({k: str(v) for k,v in source["totals"].items()})
        row.update({k: f"{v:.4f}" for k,v in source["metrics"].items()})
        output.append(row)
    return output, result


def time_text(seconds, hours=False):
    seconds = int(seconds)
    return f"{seconds//3600:02d}:{seconds//60%60:02d}:{seconds%60:02d}" if hours else f"{seconds//60:02d}:{seconds%60:02d}"


def annotation_tables(bundle, layouts):
    images, rounds = read_csv(bundle / "annotation_images.csv"), read_csv(bundle / "annotation_rounds.csv")
    if len(rounds) != 10 or sorted(int(r["round"]) for r in rounds) != list(range(1, 11)):
        raise ValueError("Expected ten counterbalanced rounds")
    groups = {key: [r for r in images if r["workflow"].removesuffix("_review") == key] for key in WORKFLOWS.values()}
    if any(len(rows) != 100 for rows in groups.values()) or len({r["image_id"] for r in images}) != 300:
        raise ValueError("Expected three disjoint 100-image timing sets")
    for key, rows in groups.items():
        if Counter(int(r["round"]) for r in rows) != Counter({i:10 for i in range(1,11)}):
            raise ValueError("Timing block membership differs")
        prefix = {"manual":"M", "sam3_exemplar_dinov2":"S", "yolo26x":"Y"}[key]
        for r in rows:
            sequence = rounds[int(r["round"])-1]
            field = ("first_block", "second_block", "third_block")[int(r["presentation_order"])-1]
            if sequence[field] != prefix + r["block"]:
                raise ValueError("Image and round presentation order disagree")
    automatic = json.loads((bundle / "automatic_processing.json").read_text())
    totals = {key: sum(int(r[key + "_block_time_seconds"]) for r in rounds) for key in groups}
    gt = {key: sum(int(r["gt_boxes"]) for r in rows) for key,rows in groups.items()}
    output = []
    for meta in layouts["relabeling_results.csv"]["rows"]:
        key = WORKFLOWS[meta["workflow"]]
        row = dict(meta)
        tp, fp, fn = [sum(int(r["reviewed_" + k]) for r in groups[key]) for k in ("tp","fp","fn")]
        seconds = automatic.get(key + "_seconds")
        row.update(n_images="100", n_gt_labels=str(gt[key]), human_time_seconds=str(totals[key]),
                   human_time_hhmmss=time_text(totals[key], True),
                   auto_label_time_seconds=str(seconds) if seconds is not None else "",
                   auto_label_time_mmss=time_text(seconds) if seconds is not None else "",
                   human_sec_per_image=f"{totals[key]/100:.2f}", human_sec_per_gt_label=f"{totals[key]/gt[key]:.2f}",
                   reduction_vs_manual_per_image=f"{1-totals[key]/totals['manual']:.3f}" if key != "manual" else "",
                   reduction_vs_manual_per_gt_label=f"{1-(totals[key]/gt[key])/(totals['manual']/gt['manual']):.3f}" if key != "manual" else "",
                   reviewed_label_f1=f"{2*tp/(2*tp+fp+fn):.4f}",
                   reviewed_label_tp=str(tp), reviewed_label_fp=str(fp), reviewed_label_fn=str(fn))
        output.append(row)
    density = []
    median = float(np.median([int(r["gt_boxes"]) for r in images]))
    for meta in layouts["annotation_density_results.csv"]["rows"]:
        row = dict(meta)
        key = WORKFLOWS[row["workflow"]]
        high = row["density_group"].lower() == "high"
        selected = [r for r in groups[key] if (int(r["gt_boxes"]) > median) == high]
        boxes = sum(int(r["gt_boxes"]) for r in selected)
        normalized = [float(r["tool_elapsed_seconds"])/int(r["gt_boxes"]) for r in selected]
        row.update(n_images=str(len(selected)), n_gt_labels=str(boxes),
                   avg_normalized_gt_box_area=f"{sum(float(r['mean_normalized_gt_box_area'])*int(r['gt_boxes']) for r in selected)/boxes:.6f}",
                   median_human_sec_per_gt_label=f"{np.median(normalized):.2f}")
        density.append(row)
    block_table = []
    for r in rounds:
        row = {k:r[k] for k in ("round", "first_block", "second_block", "third_block")}
        for key in groups:
            row[key + "_block_time_mmss"] = time_text(r[key + "_block_time_seconds"])
            row[key + "_block_time_seconds"] = r[key + "_block_time_seconds"]
        block_table.append(row)
    return {"relabeling_results.csv": output, "annotation_density_results.csv": density,
            "annotation_block_times.csv": block_table}, images, rounds


def annotation_statistics(images, rounds, protocol):
    settings = protocol["annotation_resampling"]
    indices = np.random.default_rng(settings["seed"]).integers(
        0, 10, size=(settings["round_resamples"], 10))
    x = {k:np.asarray([float(r[k+"_block_time_seconds"])/10 for r in rounds]) for k in WORKFLOWS.values()}
    contrasts, pvalues = [], []
    for key in ("sam3_exemplar_dinov2", "yolo26x"):
        diff = x["manual"] - x[key]
        boot = diff[indices].mean(axis=1)
        reduction = 1 - x[key][indices].sum(axis=1)/x["manual"][indices].sum(axis=1)
        pvalues.append(exact_sign_flip(diff))
        contrasts.append({"workflow":key, "saved_seconds_per_image":float(diff.mean()),
                          "saved_seconds_ci95":np.quantile(boot,[.025,.975]).tolist(),
                          "reduction":float(1-x[key].sum()/x["manual"].sum()),
                          "reduction_ci95":np.quantile(reduction,[.025,.975]).tolist(),
                          "raw_p":pvalues[-1]})
    for row,p in zip(contrasts,holm(pvalues)):
        row["holm_p"] = p
    # OLS with categorical round and position, manual/round1/position1 baselines.
    names = ["intercept", "sam3_exemplar_dinov2", "yolo26x"] + [f"round_{i}" for i in range(2,11)] + ["position_2", "position_3"]
    design, outcome = [], []
    for r in rounds:
        for key,prefix in (("manual","M"),("sam3_exemplar_dinov2","S"),("yolo26x","Y")):
            position = next(i for i,k in enumerate(("first_block","second_block","third_block"),1) if r[k].startswith(prefix))
            design.append([1,int(key=="sam3_exemplar_dinov2"),int(key=="yolo26x")] +
                          [int(r["round"]==str(i)) for i in range(2,11)] + [int(position==2),int(position==3)])
            outcome.append(float(r[key+"_block_time_seconds"])/10)
    X,y = np.asarray(design,dtype=float),np.asarray(outcome)
    beta,_,rank,_ = np.linalg.lstsq(X,y,rcond=None)
    df = len(y)-rank
    variance = np.square(y-X@beta).sum()/df
    se = np.sqrt(np.diag(np.linalg.inv(X.T@X))*variance)
    margin = student_t.ppf(.975,df)*se
    terms = [{"term":name,"estimate":float(b),"se":float(s),"ci95":[float(b-m),float(b+m)]} for name,b,s,m in zip(names,beta,se,margin)]
    pooled_median = float(np.median([int(r["gt_boxes"]) for r in images]))
    density = []
    block_indices = np.random.default_rng(settings["seed"]).integers(
        0, 10, size=(settings["density_resamples"], 10))
    for workflow in WORKFLOWS.values():
        source = [r for r in images if r["workflow"].removesuffix("_review") == workflow]
        for high in (False,True):
            blocks = [np.asarray([float(r["tool_elapsed_seconds"])/int(r["gt_boxes"])
                      for r in source if int(r["round"]) == i and (int(r["gt_boxes"]) > pooled_median) == high])
                      for i in range(1,11)]
            values = np.concatenate(blocks)
            samples = [np.median(np.concatenate([blocks[i] for i in ids])) for ids in block_indices
                       if sum(len(blocks[i]) for i in ids)]
            density.append({"workflow":workflow,"density":"high" if high else "low",
                            "median_seconds_per_box":float(np.median(values)),
                            "ci95":np.quantile(samples,[.025,.975]).tolist()})
    manual_medians = {r["density"]:r["median_seconds_per_box"] for r in density if r["workflow"]=="manual"}
    for r in density:
        r["reduction_from_manual"] = 1-r["median_seconds_per_box"]/manual_medians[r["density"]]
    return {"round_bootstrap_resamples":settings["round_resamples"],"round_bootstrap_seed":settings["seed"],
            "contrasts":contrasts,
            "density_bootstrap_resamples":settings["density_resamples"],"density_bootstrap_seed":settings["seed"],
            "pooled_gt_count_median":pooled_median,"density":density,
            "sensitivity":{"model":"human seconds/image ~ workflow + categorical round + categorical position",
                           "residual_df":int(df),"terms":terms}}


def adabn_selection_statistics(bundle, protocol):
    """Resample paired AP differences over training seeds and shared draws."""
    settings = protocol["adabn_setting_selection"]
    seeds = protocol["seed_order"]
    draws = list(range(settings["background_draws"]))
    budgets = (128, 512)
    expected = set(product(budgets, seeds, draws))
    cells = {}
    for row in read_csv(bundle / "adabn_selection_runs.csv"):
        key = (int(row["adaptation_images"]), int(row["training_seed"]), int(row["draw"]))
        value = float(row["ap50_95"])
        if key not in expected or key in cells or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Invalid or duplicate AdaBN selection cell")
        cells[key] = value
    if cells.keys() != expected:
        raise ValueError("Incomplete paired AdaBN selection cells")
    differences = np.asarray([
        [cells[128, seed, draw] - cells[512, seed, draw] for draw in draws]
        for seed in seeds])
    rng = np.random.default_rng(settings["bootstrap_seed"])
    seed_indices = rng.integers(0, len(seeds), (settings["bootstrap_resamples"], len(seeds)))
    draw_indices = rng.integers(0, len(draws), (settings["bootstrap_resamples"], len(draws)))
    samples = differences[seed_indices[:, :, None], draw_indices[:, None, :]].mean(axis=(1, 2))
    return {"contrast":"128 minus 512 adaptation images",
            "n_training_seeds":len(seeds), "n_background_draws":len(draws),
            "n_paired_cells":int(differences.size),
            "bootstrap_seed":settings["bootstrap_seed"],
            "bootstrap_resamples":settings["bootstrap_resamples"],
            "resampling_order":"training-seed indices, then shared-background-draw indices",
            "ap50_95_paired_effect":float(differences.mean()),
            "crossed_bootstrap_ci95":np.quantile(samples, [.025, .975], method="linear").tolist()}


def latex_escape(text):
    escaped = str(text).replace("\\",r"\textbackslash{}").replace("_",r"\_").replace("%",r"\%").replace("&",r"\&")
    # Render comparisons in math mode; OT1 text fonts map < and > to inverted punctuation.
    comparisons = {"<=": r"\(\leq\)", ">=": r"\(\geq\)", "<": r"\(<\)", ">": r"\(>\)"}
    return re.sub(r"<=|>=|<|>", lambda match: comparisons[match.group()], escaped)


def write_latex(path, outputs):
    text = [r"\documentclass{article}",r"\usepackage[a4paper,landscape,margin=15mm]{geometry}",
            r"\usepackage{booktabs,graphicx}",r"\newsavebox{\resulttable}",r"\begin{document}",r"\small"]
    headers = {"test_period":"Period", "scenario":"Scenario", "real_positive_budget":"k",
               "generation_arm":"Generation", "arm":"Arm", "precision_mean":"Precision",
               "recall_mean":"Recall", "f1_mean":"F1", "ap50_mean":"AP50", "ap50_95_mean":"AP50--95",
               "source_frame_recall_mean":"Frame recall", "false_alert_frame_fraction_mean":"False-alert frames",
               "ap50_95_paired_effect":"AP50--95 difference", "bootstrap_ci_lower":"CI lower",
               "bootstrap_ci_upper":"CI upper", "holm_adjusted_p":"Holm p", "raw_permutation_p":"Raw p",
               "two_way_bootstrap_ci_lower":"Two-way CI lower", "two_way_bootstrap_ci_upper":"Two-way CI upper",
               "n_positive_subset_means":"Positive subsets", "n_data_subset_seeds":"Subset draws",
               "n_images":"Images", "n_gt_labels":"GT boxes", "n_gt_boxes":"GT boxes",
               "human_time_seconds":"Human seconds", "human_sec_per_image":"s/image",
               "human_sec_per_gt_label":"s/box", "reduction_vs_manual_per_image":"Time reduction",
               "density_group":"Density", "avg_normalized_gt_box_area":"Mean box area",
               "median_human_sec_per_gt_label":"Median s/box", "reduction_vs_manual":"Reduction",
               "first_block":"First", "second_block":"Second", "third_block":"Third",
               "manual_block_time_seconds":"Manual (s)", "sam3_exemplar_dinov2_block_time_seconds":"SAM3 (s)",
               "yolo26x_block_time_seconds":"YOLO26x (s)"}
    labels = {"cold-start synthetic-only":"Synthetic", "synthetic-pretrain + fine-tune":"Synthetic + FT",
              "SAM3 + image exemplar + DINOv2 then human review":"SAM3 + DINOv2 + review",
              "YOLO26x then human review":"YOLO26x + review",
              "SAM3 + image exemplar + DINOv2":"SAM3 + exemplar + DINOv2"}
    for name,rows in outputs.items():
        # Keep identifiers and principal reported endpoints in the PDF; CSVs
        # retain every column, including descriptive secondary endpoints.
        fields = list(rows[0])
        if any(k.endswith("_mean") for k in fields):
            fields = [k for k in ("test_period","scenario","real_positive_budget","generation_arm","arm",
                                  "precision_mean","recall_mean","f1_mean","ap50_mean","ap50_95_mean",
                                  "source_frame_recall_mean","false_alert_frame_fraction_mean") if k in fields]
        elif "paired_effects" in name:
            fields = [k for k in ("test_period","real_positive_budget","ap50_95_paired_effect",
                                  "bootstrap_ci_lower","bootstrap_ci_upper","two_way_bootstrap_ci_lower",
                                  "two_way_bootstrap_ci_upper","n_positive_subset_means","n_data_subset_seeds",
                                  "holm_adjusted_p","raw_permutation_p") if k in fields]
        elif name == "relabeling_results.csv":
            fields = ["workflow","n_images","n_gt_labels","human_time_seconds","human_sec_per_image","human_sec_per_gt_label","reduction_vs_manual_per_image"]
        elif name == "label_recovery_results.csv":
            fields = ["method","n_images","n_gt_boxes","precision","recall","f1","tp","fp","fn"]
        elif name == "annotation_block_times.csv":
            fields = [k for k in fields if not k.endswith("_mmss")]
        text += [r"\section*{"+latex_escape(name)+"}"]
        if name == "low_budget_subset_results.csv":
            text.append(r"Metrics are means $\pm$ sample standard deviations across 15 crossed cells (five subset draws $\times$ three training seeds).\par\medskip")
        elif name == "low_budget_subset_paired_effects.csv":
            text.append(r"Intervals resample both factors; exact tests use five subset means with Holm adjustment across six comparisons.\par\medskip")
        elif any(k.endswith("_mean") for k in fields):
            text.append(r"Metrics are means $\pm$ sample standard deviations across 15 seeds.\par\medskip")
        text += [r"\begin{lrbox}{\resulttable}\begin{tabular}{"+"l"*len(fields)+"}",r"\toprule",
                 " & ".join(latex_escape(headers.get(k,k.replace("_"," "))) for k in fields)+r" \\",r"\midrule"]
        for row in rows:
            cells = []
            for k in fields:
                sd = k[:-5] + "_sample_sd"
                if k.endswith("_mean") and sd in row:
                    cells.append(f"${float(row[k]):.4f} \\pm {float(row[sd]):.4f}$")
                else:
                    cells.append(latex_escape(labels.get(row[k],row[k])))
            text.append(" & ".join(cells)+r" \\")
        text += [r"\bottomrule",r"\end{tabular}\end{lrbox}",
                 r"\noindent\ifdim\wd\resulttable>\linewidth\resizebox{.99\linewidth}{!}{\usebox{\resulttable}}\else\usebox{\resulttable}\fi",
                 r"\clearpage"]
    text.append(r"\end{document}")
    path.write_text("\n".join(text)+"\n")


def check_reference(bundle, outputs):
    comparisons = []
    for name,rows in outputs.items():
        expected = read_csv(bundle / "reference" / name)
        if len(expected) != len(rows):
            raise ValueError(f"Row count differs: {name}")
        for i,(a,b) in enumerate(zip(expected,rows),1):
            if a != b:
                diff = {k:(a.get(k),b.get(k)) for k in a.keys() | b.keys() if a.get(k)!=b.get(k)}
                raise ValueError(f"Reference mismatch {name}, row {i}: {diff}")
        comparisons.append({"table":name,"rows":len(rows),"exact_match":True})
    return comparisons


def check_display_values(bundle, output):
    checks = json.loads((bundle/"reference/paper_display_checks.json").read_text())
    files = {}
    for check in checks:
        name = check["file"]
        if name not in files:
            files[name] = read_csv(output/name) if name.endswith(".csv") else json.loads((output/name).read_text())
        value = files[name]
        for key in check["path"]:
            value = value[key]
        formatted = f"{float(value)*check['scale']:.{check['decimals']}f}"
        if formatted != check["expected"]:
            raise ValueError(f"Paper display mismatch {name} {check['path']}: {formatted} != {check['expected']}")
    return len(checks)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--check-reference",action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output}")
    args.output.mkdir(parents=True)
    protocol = json.loads((args.data/"protocol.json").read_text())
    layouts = json.loads((args.data/"table_layouts.json").read_text())
    grouped,records = detector_records(args.data,protocol)
    print(f"Validated {len(records)} fixed-main runs and recomputed deployment frame aggregation.",flush=True)
    outputs = generate_detector_tables(layouts,grouped,protocol)
    subset_cells = subset_records(args.data,protocol)
    outputs.update(generate_subset_tables(layouts,subset_cells,protocol))
    print(f"Validated {len(subset_cells)} subset-resampling runs and computed crossed-factor statistics.",flush=True)
    print("Computed detector aggregates and paired statistics.",flush=True)
    annotation,images,rounds = annotation_tables(args.data,layouts)
    outputs.update(annotation)
    timing_statistics = annotation_statistics(images,rounds,protocol)
    write_json(args.output/"annotation_statistics.json",timing_statistics)
    write_csv(args.output/"annotation_sensitivity.csv",[
        {"term":r["term"],"estimate":r["estimate"],"se":r["se"],"ci_lower":r["ci95"][0],"ci_upper":r["ci95"][1]}
        for r in timing_statistics["sensitivity"]["terms"]])
    write_csv(args.output/"annotation_density_intervals.csv",[
        {"workflow":r["workflow"],"density":r["density"],"median":r["median_seconds_per_box"],
         "ci_lower":r["ci95"][0],"ci_upper":r["ci95"][1],"reduction_from_manual":r["reduction_from_manual"]}
        for r in timing_statistics["density"]])
    print("Computed annotation summaries and round statistics.",flush=True)
    selection_statistics = adabn_selection_statistics(args.data,protocol)
    write_json(args.output/"adabn_selection_statistics.json",selection_statistics)
    print("Computed AdaBN selection interval from 300 paired-budget evaluations.",flush=True)
    label,statistics = label_results(args.data,layouts["label_recovery_results.csv"],protocol)
    outputs["label_recovery_results.csv"] = label
    write_json(args.output/"label_recovery_statistics.json",statistics)
    write_csv(args.output/"label_recovery_intervals.csv",[
        {"method":k,**r["metrics"],"f1_ci_lower":r["ci_95"]["f1"][0],"f1_ci_upper":r["ci_95"]["f1"][1]}
        for k,r in statistics["methods"].items()])
    for name,rows in outputs.items():
        write_csv(args.output/name,rows,layouts[name]["fields"] if name in layouts else None)
    write_latex(args.output/"result_tables.tex",outputs)
    comparisons = check_reference(args.data,outputs) if args.check_reference else []
    displayed = check_display_values(args.data,args.output) if args.check_reference else 0
    write_json(args.output/"verification.json",{"tables_generated":len(outputs),"reference_checks":comparisons,
        "fixed_main_runs":len(records),"subset_resampling_runs":len(subset_cells),
        "adabn_selection_runs":2*selection_statistics["n_paired_cells"],
        "paper_display_values_checked":displayed,
        "scope":protocol["scope"],"images_or_model_libraries_required":False})
    print(f"Generated {len(outputs)} result CSVs, statistical outputs and LaTeX tables in {args.output}",flush=True)
    if comparisons:
        print("Every regenerated CSV matches the retained table values exactly.",flush=True)
        print(f"Verified {displayed} displayed point estimates and interval/test values against the paper.",flush=True)


if __name__ == "__main__":
    main()
