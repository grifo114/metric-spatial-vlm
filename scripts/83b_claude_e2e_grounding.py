#!/usr/bin/env python3
"""
83b_claude_e2e_grounding.py

Standalone script to run the 5 SCI conditions on Anthropic's Claude models,
using the Anthropic API directly (not via OpenRouter).

This script is a focused complement to 83_e2e_grounding_test_official_v2.py.
It reuses the same benchmark CSVs, the same pre-rendered scene images
(under artifacts/e2e_grounding_renders/), and writes CSVs in the SAME format
as the v2 script so they can be loaded and aggregated by the same analysis
code without changes.

Key differences vs the v2 script:
  - max_tokens is much higher (2048 by default; 4096 if --reasoning).
    Claude tends to think out loud; 100 tokens (the v2 default) is far too
    little.
  - The prompt explicitly asks Claude to put the final answer on the
    FIRST LINE, so even if the budget is exhausted by reasoning later, the
    answer is captured.
  - Uses the official `anthropic` Python SDK.

==============================================================
Setup (one time)
==============================================================
    pip install anthropic pandas pyvista pillow plyfile scipy numpy

==============================================================
Usage examples (run from the repo root)
==============================================================

    # Smoke test: 5 queries, prints prompts, no API call
    python scripts/83b_claude_e2e_grounding.py \\
        --prompt-mode original --dry-run 5 --print-prompts --no-api

    # Real run: Baseline (original prompt)
    ANTHROPIC_API_KEY=sk-ant-... python scripts/83b_claude_e2e_grounding.py \\
        --prompt-mode original --language en --operator distance \\
        --output-suffix _distance_baseline_en_claude_sonnet45_run1

    # Real run: SCI L1
    ANTHROPIC_API_KEY=sk-ant-... python scripts/83b_claude_e2e_grounding.py \\
        --prompt-mode context --descriptor-level 1 --language en --operator distance \\
        --output-suffix _distance_ctx_l1_en_claude_sonnet45_run1

The full matrix is 5 conditions x N runs (set N=1 to match the Gemini setup,
or N=2 to match the other models).
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd

# anthropic is an optional import: only required when actually calling the API
try:
    import anthropic  # type: ignore
except ImportError:
    anthropic = None  # type: ignore

# ---------------------------------------------------------------------------
# Path setup — same as v2 script so it Just Works from the repo root
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.grounding.prompt_enrichment import (  # noqa: E402
    excluded_for_query,
    format_object_list,
    format_object_list_with_context,
)
from src.grounding.example_selection import pick_example_ids  # noqa: E402

# ---------------------------------------------------------------------------
# Geometry helpers — duplicated locally so this script is self-contained
# ---------------------------------------------------------------------------
import numpy as np
from scipy.spatial import cKDTree

BENCHMARK    = ROOT / "benchmark"
POINTS_DIR   = ROOT / "artifacts" / "object_points_test_official_stage1"
RENDERS_DIR  = ROOT / "artifacts" / "e2e_grounding_renders"
RESULTS_DIR  = ROOT / "results" / "benchmark_v1"

GT_CSV       = BENCHMARK / "ground_truth_distance_nearest_test_official_stage1.csv"
QUERIES_CSV  = BENCHMARK / "queries_test_official_stage1_distance_nearest_final.csv"
MANIFEST_CSV = BENCHMARK / "objects_manifest_test_official_stage1.csv"


def load_points(obj_id: str, scene_id: str):
    path = POINTS_DIR / scene_id / f"{obj_id}.npz"
    if not path.exists():
        return None
    return np.load(path)["points"]


def surface_distance(pts_a, pts_b) -> float:
    tree = cKDTree(pts_b)
    dists, _ = tree.query(pts_a, k=1, workers=-1)
    return float(dists.min())


def centroid_distance(pts_a, pts_b) -> float:
    return float(np.linalg.norm(pts_a.mean(axis=0) - pts_b.mean(axis=0)))


# ---------------------------------------------------------------------------
# Number map — same logic as v2 (sort by category, then object_id)
# ---------------------------------------------------------------------------
def build_number_map(scene_df: pd.DataFrame):
    rows = scene_df.sort_values(["label_norm", "object_id"]).reset_index(drop=True)
    num_to_id, id_to_num = {}, {}
    for i, row in rows.iterrows():
        n = i + 1
        oid = row["object_id"]
        num_to_id[n] = oid
        id_to_num[oid] = n
    return num_to_id, id_to_num


# ---------------------------------------------------------------------------
# Prompts — adapted for Claude so the FINAL ANSWER is the FIRST LINE.
# This way, even if the model spends its budget reasoning afterwards, the
# parser still finds the answer.
# ---------------------------------------------------------------------------

CLAUDE_DISTANCE_PROMPT_EN = """\
You are looking at a top-down view of a 3D indoor scene.
Each object is marked with a number (colored circle).

Objects in the scene:
{object_list}

The query asks for the distance between two objects:
- A: a {label_a}
- B: a {label_b}

CRITICAL: Your response must START with the answer on the very first line,
in this exact format (two object_ids separated by comma, no other text on
the first line):
{example_a}, {example_b}

After the first line you may optionally explain your reasoning, but the
first line MUST be only the two ids. If multiple instances of the same
category exist in the scene, choose the most prominent or contextually
meaningful one.
"""


def get_distance_prompt(language: str) -> str:
    # Only EN provided here; matches the language used for all multi-VLM runs
    if language != "en":
        raise NotImplementedError(
            "This script is set up for --language en only, matching the "
            "multi-VLM experimental setup in Section 4 of the paper."
        )
    return CLAUDE_DISTANCE_PROMPT_EN


# ---------------------------------------------------------------------------
# Anthropic API call — uses the official SDK, with optional retry
# ---------------------------------------------------------------------------
MAX_RETRIES = 3
RETRY_DELAY = 6


def call_claude(
    client,
    model: str,
    prompt: str,
    img_path: Path,
    max_tokens: int,
    temperature: float = 0.0,
) -> str:
    """Send a multimodal message to Claude and return the raw text reply."""
    with open(img_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )

    parts = []
    for block in message.content:
        # Skip thinking blocks (extended thinking, if ever enabled).
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "\n".join(parts).strip()


def call_with_retry(client, model, prompt, img_path, max_tokens, temperature):
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            return call_claude(client, model, prompt, img_path, max_tokens, temperature)
        except Exception as e:
            last_err = str(e)
            print(f"  [attempt {attempt+1}/{MAX_RETRIES}] {last_err[:300]}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
    raise RuntimeError(f"API failed after {MAX_RETRIES} attempts: {last_err}")


# ---------------------------------------------------------------------------
# Response parsing.
#
# Claude is asked to put the answer on the FIRST LINE in the format:
#   scene_X__cat_NNN, scene_X__cat_NNN
# We extract from that line first; if no valid IDs are found there, we fall
# back to a full-response sweep to be robust against minor format slips.
# ---------------------------------------------------------------------------

def extract_ids(
    response: str,
    valid_ids: set,
    n: int,
    num_to_id: dict | None = None,
):
    """Return a list of exactly n object ids, padding with None as needed."""
    found = []

    # First pass: just the first non-empty line (this is what Claude is asked
    # to put the answer in).
    first_line = ""
    for ln in response.splitlines():
        if ln.strip():
            first_line = ln.strip()
            break

    def _scan(text: str):
        # full ids
        for token in re.split(r"[\s,;]+", text):
            token = token.strip().strip(".,;\"'*`")
            if token in valid_ids and token not in found:
                found.append(token)
            if len(found) == n:
                return
        # bare numbers
        if num_to_id is not None and len(found) < n:
            for token in re.split(r"[\s,;]+", text):
                token = token.strip().strip(".,;\"'*`")
                if token.isdigit():
                    oid = num_to_id.get(int(token))
                    if oid and oid in valid_ids and oid not in found:
                        found.append(oid)
                if len(found) == n:
                    return

    _scan(first_line)
    if len(found) < n:
        # Fallback: scan the whole response in case the first-line format slipped
        _scan(response)

    while len(found) < n:
        found.append(None)
    return found[:n]


# ---------------------------------------------------------------------------
# IO helpers — same column schema as the v2 script for downstream compatibility
# ---------------------------------------------------------------------------

def _save(path: Path, data: dict) -> None:
    df = pd.DataFrame([data])
    if path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, mode="w", header=True, index=False)


def _err_row(qid, scene_id, operator, err) -> dict:
    return {
        "query_id": qid, "scene_id": scene_id, "operator": operator,
        "provider": "anthropic", "model": None,
        "prompt_mode": None, "descriptor_level": None,
        "context_scope": None, "language": None,
        "example_a": None, "example_b": None, "example_ref": None,
        "gt_value": None, "gt_object_a": None, "gt_object_b": None,
        "grounded_a": None, "grounded_b": None,
        "grounding_correct": None,
        "e_total_surface": None, "e_total_centroid": None,
        "nearest_vlm_answer": None, "nearest_vlm_dist": None,
        "vlm_response": "ERROR", "error": err,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", type=int, default=0)
    parser.add_argument("--operator", choices=["distance", "nearest", "all"],
                        default="distance",
                        help="This standalone Claude runner focuses on "
                             "distance, matching the multi-VLM setup. Set to "
                             "'all' if you ever extend it to other operators.")
    parser.add_argument("--prompt-mode", choices=["original", "context"],
                        default="original")
    parser.add_argument("--descriptor-level", type=int, choices=[1, 2, 3],
                        default=2)
    parser.add_argument("--context-scope", choices=["all", "reference_only"],
                        default="all")
    parser.add_argument("--language", choices=["en"], default="en",
                        help="EN only, matching the multi-VLM setup.")
    parser.add_argument("--model", default="claude-sonnet-4-5",
                        help="Anthropic model identifier. Use the exact "
                             "snapshot string from the Anthropic console, "
                             "e.g. 'claude-sonnet-4-5' or "
                             "'claude-sonnet-4-5-20250929'.")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="Output token budget. 2048 is comfortable for "
                             "Claude's reasoning style; raise to 4096 if you "
                             "see truncated responses.")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Default 0.0 for reproducibility. Set to >0 if "
                             "you want to replicate the non-deterministic "
                             "behaviour of the OpenAI/OpenRouter runs.")
    parser.add_argument("--output-suffix", default="",
                        help="Suffix appended to the output CSV name.")
    parser.add_argument("--print-prompts", action="store_true")
    parser.add_argument("--no-api", action="store_true",
                        help="Skip API calls. Useful for prompt inspection.")
    parser.add_argument("--scene-id", default=None)
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not args.no_api:
        if anthropic is None:
            sys.exit("anthropic SDK not installed. Run: pip install anthropic")
        if not api_key:
            sys.exit("ANTHROPIC_API_KEY env var is not set.")

    client = None
    if not args.no_api:
        client = anthropic.Anthropic(api_key=api_key)

    output_csv = RESULTS_DIR / f"e2e_grounding_test_official_raw{args.output_suffix}.csv"

    print("== Config ==")
    print(f"  provider         : anthropic")
    print(f"  model            : {args.model}")
    print(f"  prompt_mode      : {args.prompt_mode}")
    print(f"  descriptor_level : {args.descriptor_level}")
    print(f"  context_scope    : {args.context_scope}")
    print(f"  language         : {args.language}")
    print(f"  operator         : {args.operator}")
    print(f"  max_tokens       : {args.max_tokens}")
    print(f"  temperature      : {args.temperature}")
    print(f"  output           : {output_csv.name}")
    if args.no_api:
        print(f"  API CALLS        : DISABLED (--no-api)")
    print()

    # ---------------------------------------------------------------------
    # Load data — same logic as v2
    # ---------------------------------------------------------------------
    gt_df       = pd.read_csv(GT_CSV)
    queries_df  = pd.read_csv(QUERIES_CSV)
    manifest_df = pd.read_csv(MANIFEST_CSV)

    queries_df = queries_df[queries_df["review_keep"] == "yes"].copy()

    extra = ["scene_id", "operator", "structured_query",
             "label_a", "label_b",
             "reference_object", "reference_label", "target_category",
             "answer_object"]
    for c in extra:
        if c not in queries_df.columns:
            queries_df[c] = None

    merged = gt_df.merge(
        queries_df[extra].drop_duplicates("structured_query"),
        on=["scene_id", "operator", "structured_query"], how="left",
    )

    if args.operator != "all":
        merged = merged[merged["operator"] == args.operator].copy()
    if args.scene_id is not None:
        merged = merged[merged["scene_id"] == args.scene_id].copy()
    if args.dry_run > 0:
        merged = merged.head(args.dry_run).copy()

    print(f"Queries: {len(merged)}")

    # Resume support — skip queries that are already done in the output CSV.
    done_ids = set()
    if output_csv.exists() and not args.no_api:
        done_ids = set(pd.read_csv(output_csv)["query_id"].tolist())
        print(f"Resuming: {len(done_ids)} already done.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Scene cache: holds (scene_df, num_to_id, id_to_num, render_path).
    scene_cache = {}

    def get_scene_data(scene_id: str):
        if scene_id not in scene_cache:
            sdf = manifest_df[
                (manifest_df["scene_id"] == scene_id) &
                (manifest_df["is_valid_object"] == True)
            ].copy()
            num_to_id, id_to_num = build_number_map(sdf)
            render_path = RENDERS_DIR / f"{scene_id}_numbered.jpg"
            if not args.no_api and not render_path.exists():
                # We expect renders to already exist from prior v2 runs. If
                # they don't, the user should run the v2 script first (or
                # add render generation here).
                raise FileNotFoundError(
                    f"Render not found: {render_path}. "
                    f"Run the v2 script first to generate it."
                )
            scene_cache[scene_id] = (sdf, num_to_id, id_to_num, render_path)
        return scene_cache[scene_id]

    n_ok, n_err = 0, 0

    for _, row in merged.iterrows():
        qid      = str(row["query_id"])
        scene_id = str(row["scene_id"])
        operator = str(row["operator"])

        if qid in done_ids:
            continue

        print(f"\n{qid} | {operator} | {scene_id}")

        try:
            sdf, num_to_id, id_to_num, render_path = get_scene_data(scene_id)
        except Exception as e:
            print(f"  ERROR scene: {e}")
            _save(output_csv, _err_row(qid, scene_id, operator, str(e)))
            n_err += 1
            continue

        valid_ids = set(sdf["object_id"].tolist())

        # ---- Build object list ----
        if args.prompt_mode == "original":
            obj_list = format_object_list(num_to_id, sdf)
        else:
            excluded = excluded_for_query(
                operator,
                label_a=row.get("label_a"),
                label_b=row.get("label_b"),
                reference_label=row.get("reference_label"),
                target_category=row.get("target_category"),
            )
            if args.context_scope == "all":
                obj_list = format_object_list_with_context(
                    num_to_id, sdf,
                    level=args.descriptor_level,
                    excluded_categories=excluded,
                    language=args.language,
                    scope_category=None,
                )
            else:  # reference_only — distance: scope = both query categories
                label_a_q = str(row.get("label_a") or "")
                label_b_q = str(row.get("label_b") or "")
                list_a = format_object_list_with_context(
                    num_to_id, sdf,
                    level=args.descriptor_level,
                    excluded_categories=excluded,
                    language=args.language,
                    scope_category=label_a_q,
                ).splitlines()
                list_b = format_object_list_with_context(
                    num_to_id, sdf,
                    level=args.descriptor_level,
                    excluded_categories=excluded,
                    language=args.language,
                    scope_category=label_b_q,
                ).splitlines()
                merged_lines = []
                for la, lb in zip(list_a, list_b):
                    merged_lines.append(la if "(" in la else lb)
                obj_list = "\n".join(merged_lines)

        # ---- Distance prompt assembly ----
        if operator != "distance":
            # This script focuses on distance to match the multi-VLM setup.
            # If you ever extend it, add nearest handling here.
            continue

        label_a  = str(row.get("label_a") or "")
        label_b  = str(row.get("label_b") or "")
        gt_obj_a = str(row["gt_object_a"])
        gt_obj_b = str(row["gt_object_b"])
        gt_val   = float(row["gt_distance_m"])

        example_a, example_b = pick_example_ids(
            sdf, "distance",
            gt_object_a=gt_obj_a, gt_object_b=gt_obj_b,
            label_a=label_a, label_b=label_b,
            seed_key=qid,
        )
        if example_a is None or example_b is None:
            ids = sorted(valid_ids)
            example_a = ids[0] if ids else "OBJECT_A"
            example_b = ids[1] if len(ids) > 1 else "OBJECT_B"

        prompt = get_distance_prompt(args.language).format(
            object_list=obj_list, label_a=label_a, label_b=label_b,
            example_a=example_a, example_b=example_b,
        )

        if args.print_prompts:
            print("  ----- PROMPT -----")
            for ln in prompt.splitlines():
                print(f"  | {ln}")
            print("  ----- END PROMPT -----")

        if args.no_api:
            n_ok += 1
            continue

        # ---- Claude API call ----
        try:
            response = call_with_retry(
                client, args.model, prompt, render_path,
                max_tokens=args.max_tokens, temperature=args.temperature,
            )
            preview = response[:120].replace("\n", " ⏎ ")
            print(f"  {args.model} → {preview!r}")
        except Exception as e:
            print(f"  ERROR API: {e}")
            _save(output_csv, _err_row(qid, scene_id, operator, f"API_ERROR: {e}"))
            n_err += 1
            time.sleep(RETRY_DELAY)
            continue

        # ---- Parse + grounding check ----
        ids = extract_ids(response, valid_ids, 2, num_to_id=num_to_id)
        grounded_a, grounded_b = ids[0], ids[1]
        grounding_correct = (
            (grounded_a == gt_obj_a and grounded_b == gt_obj_b) or
            (grounded_a == gt_obj_b and grounded_b == gt_obj_a)
        )

        e_surf = None
        e_cent = None
        if grounded_a and grounded_b:
            pts_a = load_points(grounded_a, scene_id)
            pts_b = load_points(grounded_b, scene_id)
            if pts_a is not None and pts_b is not None:
                e_surf = surface_distance(pts_a, pts_b)
                e_cent = centroid_distance(pts_a, pts_b)

        result = {
            "query_id": qid, "scene_id": scene_id, "operator": operator,
            "provider": "anthropic", "model": args.model,
            "prompt_mode": args.prompt_mode,
            "descriptor_level": args.descriptor_level if args.prompt_mode == "context" else None,
            "context_scope": args.context_scope if args.prompt_mode == "context" else None,
            "language": args.language,
            "example_a": example_a, "example_b": example_b,
            "example_ref": None,
            "gt_value": gt_val,
            "gt_object_a": gt_obj_a, "gt_object_b": gt_obj_b,
            "grounded_a": grounded_a, "grounded_b": grounded_b,
            "grounding_correct": grounding_correct,
            "e_total_surface": e_surf,
            "e_total_centroid": e_cent,
            "nearest_vlm_answer": None,
            "nearest_vlm_dist":   None,
            "vlm_response": response, "error": None,
        }

        _save(output_csv, result)
        n_ok += 1
        print(f"  grounding_correct={grounding_correct}")
        time.sleep(0.4)

    print(f"\n{'='*60}")
    print(f"Done: {n_ok} OK | {n_err} errors")
    if not args.no_api:
        print(f"Results: {output_csv}")


if __name__ == "__main__":
    main()