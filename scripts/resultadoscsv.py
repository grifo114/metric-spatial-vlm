import pandas as pd

manifest = pd.read_csv("benchmark/objects_manifest_test_official_stage1.csv")
manifest = manifest[manifest["is_valid_object"] == True]

gt = pd.read_csv("benchmark/ground_truth_distance_nearest_test_official_stage1.csv")
queries = pd.read_csv("benchmark/queries_test_official_stage1_distance_nearest_final.csv")
queries = queries[(queries["operator"] == "nearest") & (queries["review_keep"] == "yes")]
gt_n = gt[gt["operator"] == "nearest"][["query_id", "structured_query"]]
queries = queries.merge(gt_n, on="structured_query", how="inner")

def n_refs(row):
    return len(manifest[(manifest["scene_id"] == row["scene_id"]) &
                        (manifest["label_norm"] == row["reference_label"])])
queries["n_refs"] = queries.apply(n_refs, axis=1)

base = pd.read_csv("results/benchmark_v1/e2e_grounding_test_official_raw_nearest_baseline_v2.csv")
ctx_all = pd.read_csv("results/benchmark_v1/e2e_grounding_test_official_raw_nearest_ctx_l2.csv")
ctx_ref = pd.read_csv("results/benchmark_v1/e2e_grounding_test_official_raw_nearest_ctx_l2_ref_only.csv")

meta = queries[["query_id", "n_refs", "target_category", "reference_label"]]

def merge_in(df):
    return df.merge(meta, on="query_id", how="left")

base, ctx_all, ctx_ref = merge_in(base), merge_in(ctx_all), merge_in(ctx_ref)

print("=== Stratified grounding rate ===")
print(f"{'Stratum':<25} {'Baseline':>10} {'CtxAll':>10} {'CtxRefOnly':>12} {'n':>4}")
for label, mask in [
    ("trivial (n_refs=1)",  base["n_refs"] == 1),
    ("informative (n>=2)",  base["n_refs"] >= 2),
    ("hard (n_refs>=5)",    base["n_refs"] >= 5),
    ("overall",             base["n_refs"] >= 1),
]:
    b = base[mask]["grounding_correct"].mean()
    a = ctx_all[mask]["grounding_correct"].mean()
    r = ctx_ref[mask]["grounding_correct"].mean()
    n = mask.sum()
    print(f"{label:<25} {b:>9.1%}  {a:>9.1%}  {r:>11.1%}  {n:>4}")

# Per-query 3-way comparison — keep meta columns explicitly
m = base[["query_id", "n_refs", "target_category", "reference_label",
          "grounding_correct", "grounded_a"]].rename(
    columns={"grounding_correct": "base_ok", "grounded_a": "base_pred"}
).merge(
    ctx_all[["query_id", "grounding_correct", "grounded_a"]].rename(
        columns={"grounding_correct": "all_ok", "grounded_a": "all_pred"}),
    on="query_id"
).merge(
    ctx_ref[["query_id", "grounding_correct", "grounded_a"]].rename(
        columns={"grounding_correct": "ref_ok", "grounded_a": "ref_pred"}),
    on="query_id"
)

# Type A: n_refs=1, target=chair (the catastrophic 5 from ctx_all)
type_a = m[(m["n_refs"] == 1) & (m["target_category"] == "chair")]
print()
print("=== Type A queries (n_refs=1, target=chair) ===")
print(type_a[["query_id", "reference_label", "base_ok", "all_ok", "ref_ok",
              "base_pred", "all_pred", "ref_pred"]].to_string(index=False))

# Hard queries: where ref_only doubled accuracy
hard = m[m["n_refs"] >= 5]
print()
print("=== Hard queries (n_refs>=5) — where ref_only helped most ===")
print(hard[["query_id", "n_refs", "reference_label", "target_category",
            "base_ok", "all_ok", "ref_ok"]].to_string(index=False))

# Hard transitions
print()
print("=== Hard query transitions: baseline vs ctx_ref_only ===")
print(f"  False → True (ref_only gain) : {((~hard['base_ok']) & ( hard['ref_ok'])).sum()}")
print(f"  True → False (ref_only loss) : {(( hard['base_ok']) & (~hard['ref_ok'])).sum()}")
print(f"  Both correct                 : {(( hard['base_ok']) & ( hard['ref_ok'])).sum()}")
print(f"  Both wrong                   : {((~hard['base_ok']) & (~hard['ref_ok'])).sum()}")

# Where ref_only predicted in target_category instead of reference
def cat_of(oid):
    if not isinstance(oid, str): return None
    row = manifest[manifest["object_id"] == oid]
    return row["label_norm"].iloc[0] if not row.empty else None

m["all_cat"] = m["all_pred"].apply(cat_of)
m["ref_cat"] = m["ref_pred"].apply(cat_of)

confused_all = m[(m["all_cat"] == m["target_category"]) & (m["all_cat"] != m["reference_label"])]
confused_ref = m[(m["ref_cat"] == m["target_category"]) & (m["ref_cat"] != m["reference_label"])]
print()
print(f"=== Category confusion ===")
print(f"  Context ALL      : {len(confused_all)}/39")
print(f"  Context REF_ONLY : {len(confused_ref)}/39")