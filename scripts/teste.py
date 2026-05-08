import pandas as pd
queries = pd.read_csv("benchmark/queries_test_official_stage1_distance_nearest_final.csv")
queries = queries[(queries["operator"] == "distance") & (queries["review_keep"] == "yes")]

same_cat = queries[queries["label_a"] == queries["label_b"]]
print(f"Distance queries with label_a == label_b: {len(same_cat)}/{len(queries)}")
if len(same_cat) > 0:
    print(same_cat[["scene_id", "label_a", "label_b", "object_a", "object_b"]].to_string())

# Also check: in cases where label_a appears N times in scene, how often is the GT 
# a specific instance among them?
print("\n=== Multi-instance label_a cases ===")
manifest = pd.read_csv("benchmark/objects_manifest_test_official_stage1.csv")
manifest = manifest[manifest["is_valid_object"] == True]

multi_cases = []
for _, q in queries.iterrows():
    n_a = len(manifest[(manifest["scene_id"]==q["scene_id"]) & (manifest["label_norm"]==q["label_a"])])
    n_b = len(manifest[(manifest["scene_id"]==q["scene_id"]) & (manifest["label_norm"]==q["label_b"])])
    if n_a >= 2 or n_b >= 2:
        multi_cases.append((q["label_a"], n_a, q["label_b"], n_b))

print(f"Queries where at least one category has >=2 instances: {len(multi_cases)}/{len(queries)}")
print(f"Distribution of (n_a, n_b) pairs:")
from collections import Counter
counts = Counter((min(a,b), max(a,b)) for _, a, _, b in multi_cases)
for k, v in sorted(counts.items()):
    print(f"  ({k[0]}, {k[1]}): {v}")