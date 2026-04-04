"""
09_language_grounding_llava.py  v2

Fix: extract_json now handles markdown code blocks (```json ... ```)
which LLaVA 7b wraps around its JSON responses.
"""

import os, json, re
import numpy as np
import pandas as pd
import ollama
from tqdm import tqdm

INPUT_CSV   = "results/selected_pairs.csv"
GT_JSON     = "results/gt_centroids.json"
RGBD_CSV    = "results/predictions_proposed_rgbd.csv"

MODELS = ["llava:7b", "llava:13b"]

SYSTEM_PROMPT = """You are a spatial reasoning assistant.
You will receive a natural language query about objects in an indoor scene
and a list of objects with their IDs and labels.

Your task: identify exactly which two objects the query refers to.

Return ONLY a JSON object like this:
{"object_a": <integer_id>, "object_b": <integer_id>}

No explanation. No extra text. Only the JSON."""

def build_prompt(query, objects):
    obj_list = "\n".join(
        f"  ID {o['object_id']}: {o['label']}"
        for o in objects
        if o.get("in_target", True)
    )
    return f'Query: "{query}"\n\nObjects in this scene:\n{obj_list}\n\nReturn JSON only.'

def extract_json(text):
    text = text.replace("object\\_a", "object_a").replace("object\\_b", "object_b")
    """Extract JSON from LLaVA response — handles plain JSON and markdown blocks."""
    # Remove markdown code blocks: ```json ... ``` or ``` ... ```
    cleaned = re.sub(r'```(?:json)?\s*(.*?)\s*```', r'\1', text, flags=re.DOTALL)
    cleaned = cleaned.replace('\\_', '_')
    # Try direct parse
    try:
        return json.loads(cleaned.strip())
    except Exception:
        pass

    # Try to find first JSON object in text
    match = re.search(r'\{[^}]+\}', cleaned)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass

    return None

def call_llava(model, query, objects, retries=3):
    prompt = build_prompt(query, objects)
    for attempt in range(retries):
        try:
            response = ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                options={"temperature": 0},
            )
            content = response["message"]["content"]
            parsed  = extract_json(content)
            if parsed is None:
                raise ValueError(f"No JSON found: {repr(content[:200])}")
            id_a = int(parsed.get("object_a") or parsed.get("object_id_a"))
            id_b = int(parsed.get("object_b") or parsed.get("object_id_b"))
            return id_a, id_b
        except Exception:
            if attempt == retries - 1:
                return None, None
    return None, None

def run_model(model_name, df, gt_data, rgbd_lookup):
    slug = model_name.replace(":", "").replace(".", "")
    out_grounding = f"results/grounding_{slug}.csv"
    out_e2e       = f"results/e2e_{slug}.csv"

    print(f"\n{'='*55}")
    print(f"Model: {model_name}")
    print(f"{'='*55}")

    grounding_results = []
    e2e_results       = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=model_name):
        scene_id  = row["scene_id"]
        query     = row["query"]
        gt_id_a   = int(row["object_id_a"])
        gt_id_b   = int(row["object_id_b"])
        pair_key  = row["pair_key"]

        scene_objects = gt_data.get(scene_id, [])
        obj_lookup    = {o["object_id"]: o for o in scene_objects}

        pred_id_a, pred_id_b = call_llava(model_name, query, scene_objects)

        if pred_id_a is None or pred_id_b is None:
            grounding_status = "parse_error"
            id_correct       = False
        else:
            id_correct       = ({pred_id_a, pred_id_b} == {gt_id_a, gt_id_b})
            grounding_status = "ok"

        grounding_results.append({
            "scene_id":   scene_id,
            "query":      query,
            "gt_id_a":    gt_id_a,
            "gt_id_b":    gt_id_b,
            "pred_id_a":  pred_id_a,
            "pred_id_b":  pred_id_b,
            "id_correct": id_correct,
            "range":      row["range"],
            "label_a":    row["label_a"],
            "label_b":    row["label_b"],
            "status":     grounding_status,
            "model":      model_name,
        })

        # End-to-end
        rgbd_info            = rgbd_lookup.get(pair_key, (None, "not_found"))
        rgbd_dist, rgbd_status = rgbd_info

        if grounding_status != "ok":
            e2e_dist, e2e_status = None, "grounding_parse_error"
        elif id_correct and rgbd_status == "ok":
            e2e_dist, e2e_status = rgbd_dist, "ok"
        elif id_correct and rgbd_status != "ok":
            e2e_dist, e2e_status = None, "rgbd_failed"
        else:
            if pred_id_a in obj_lookup and pred_id_b in obj_lookup:
                ca = np.array(obj_lookup[pred_id_a]["centroid"])
                cb = np.array(obj_lookup[pred_id_b]["centroid"])
                e2e_dist   = round(float(np.linalg.norm(ca - cb)), 4)
                e2e_status = "grounding_error"
            else:
                e2e_dist, e2e_status = None, "grounding_error_unknown_obj"

        e2e_results.append({
            "scene_id":        scene_id,
            "range":           row["range"],
            "gt_distance_m":   row["gt_distance_m"],
            "pred_distance_m": e2e_dist,
            "label_a":         row["label_a"],
            "label_b":         row["label_b"],
            "id_correct":      id_correct,
            "model":           model_name,
            "status":          e2e_status,
        })

    gr_df  = pd.DataFrame(grounding_results)
    e2e_df = pd.DataFrame(e2e_results)
    gr_df.to_csv(out_grounding, index=False)
    e2e_df.to_csv(out_e2e,      index=False)

    # Summary
    valid_gr = gr_df[gr_df["status"] == "ok"]
    acc      = valid_gr["id_correct"].mean() * 100 if len(valid_gr) > 0 else 0

    ok_e2e = e2e_df[e2e_df["status"] == "ok"].copy()
    ok_e2e["abs_error"] = (ok_e2e["pred_distance_m"] - ok_e2e["gt_distance_m"]).abs()
    ok_e2e["rel_error"] = ok_e2e["abs_error"] / ok_e2e["gt_distance_m"] * 100

    all_e2e = e2e_df[e2e_df["pred_distance_m"].notna()].copy()
    all_e2e["abs_error"] = (all_e2e["pred_distance_m"] - all_e2e["gt_distance_m"]).abs()
    all_e2e["rel_error"] = all_e2e["abs_error"] / all_e2e["gt_distance_m"] * 100

    print(f"\n{'─'*55}")
    print(f"{model_name} — Results")
    print(f"{'─'*55}")
    print(f"Pairs processed      : {len(gr_df)}")
    print(f"Parse errors         : {(gr_df['status'] != 'ok').sum()}")
    print(f"ID accuracy          : {acc:.1f}%")
    print(f"  Correct pairs      : {int(valid_gr['id_correct'].sum())}/{len(valid_gr)}")

    print(f"\nEnd-to-end (grounding correct + RGBD ok):")
    print(f"  Valid pairs        : {len(ok_e2e)}")
    if len(ok_e2e) > 0:
        print(f"  MAE                : {ok_e2e['abs_error'].mean():.4f} m")
        print(f"  Mean Rel. Error    : {ok_e2e['rel_error'].mean():.2f} %")
        print(f"\n  By range:")
        for rng in ["short", "medium", "long"]:
            sub = ok_e2e[ok_e2e["range"] == rng]
            if len(sub) == 0:
                continue
            print(f"    {rng:<8}  n={len(sub):>3}  "
                  f"MAE={sub['abs_error'].mean():.3f} m  "
                  f"RelErr={sub['rel_error'].mean():.1f}%")

    if len(all_e2e) > 0:
        print(f"\nEnd-to-end (all pairs with prediction):")
        print(f"  n={len(all_e2e)}  "
              f"MAE={all_e2e['abs_error'].mean():.4f} m  "
              f"RelErr={all_e2e['rel_error'].mean():.2f}%")

    print(f"\nStatus breakdown (e2e):")
    print(e2e_df["status"].value_counts().to_string())

    return {
        "model":        model_name,
        "acc":          round(acc, 1),
        "parse_errors": int((gr_df["status"] != "ok").sum()),
        "e2e_valid":    len(ok_e2e),
        "MAE":          round(ok_e2e["abs_error"].mean(), 4) if len(ok_e2e) > 0 else None,
        "RelErr":       round(ok_e2e["rel_error"].mean(), 2) if len(ok_e2e) > 0 else None,
    }

def main():
    df      = pd.read_csv(INPUT_CSV)
    rgbd_df = pd.read_csv(RGBD_CSV)

    with open(GT_JSON) as f:
        gt_data = json.load(f)

    df["pair_key"] = (
        df["scene_id"] + "_" +
        df["object_id_a"].astype(str) + "_" +
        df["object_id_b"].astype(str)
    )
    rgbd_df["pair_key"] = (
        rgbd_df["scene_id"] + "_" +
        df["object_id_a"].astype(str) + "_" +
        df["object_id_b"].astype(str)
    )
    rgbd_lookup = dict(zip(
        rgbd_df["pair_key"],
        zip(rgbd_df["pred_distance_m"], rgbd_df["status"])
    ))

    all_summaries = []
    for model_name in MODELS:
        summary = run_model(model_name, df, gt_data, rgbd_lookup)
        all_summaries.append(summary)

    print(f"\n{'='*55}")
    print(f"COMPARISON — Language Grounding Models")
    print(f"{'='*55}")
    print(f"{'Model':<20} {'Acc':>6} {'ParseErr':>9} {'E2E_n':>6} {'MAE':>8} {'RelErr':>8}")
    print("─" * 55)
    for s in all_summaries:
        print(f"{s['model']:<20} {s['acc']:>5.1f}% {s['parse_errors']:>9} "
              f"{s['e2e_valid']:>6} {str(s['MAE']):>8} {str(s['RelErr']):>7}%")

if __name__ == "__main__":
    main()
