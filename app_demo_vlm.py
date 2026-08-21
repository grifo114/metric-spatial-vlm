import sys
import importlib.util
from pathlib import Path

import pandas as pd
import streamlit as st
from src.demo_visuals import draw_topdown_overlay, render_scene_3d_static, find_perspective_image, draw_perspective_overlay


# ============================================================
# Basic configuration
# ============================================================

ROOT = Path(__file__).resolve().parent
SCRIPT_PATH = ROOT / "scripts" / "83_e2e_grounding_test_official_v2.py"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("e2e_script", SCRIPT_PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

API_ENDPOINTS_LOCAL = {
    "openai": "https://api.openai.com/v1/chat/completions",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
}

OPENAI_MODELS = [
    "gpt-4.1",
    "gpt-4.1-mini",
]

OPENROUTER_MODELS = [
    "qwen/qwen3-vl-8b-instruct",
    "qwen/qwen3-vl-32b-instruct",
    "qwen/qwen3-vl-235b-a22b-instruct",
]


# ============================================================
# Local helper functions
# ============================================================

def pick_example_ids_local(
    sdf,
    operator,
    gt_object_a=None,
    gt_object_b=None,
    label_a=None,
    label_b=None,
    seed_key=None,
):
    """Select two valid object IDs to use as examples in the prompt."""

    if sdf is None or sdf.empty:
        return None, None

    valid_ids = set(sdf["object_id"].dropna().astype(str).tolist())

    if len(valid_ids) < 2:
        ids = sorted(valid_ids)
        if len(ids) == 1:
            return ids[0], None
        return None, None

    forbidden = {str(gt_object_a), str(gt_object_b)}

    candidates = sdf.copy()
    candidates["object_id"] = candidates["object_id"].astype(str)

    if "label_norm" in candidates.columns:
        a_candidates = candidates[
            (candidates["label_norm"].astype(str) == str(label_a))
            & (~candidates["object_id"].isin(forbidden))
        ]["object_id"].tolist()

        b_candidates = candidates[
            (candidates["label_norm"].astype(str) == str(label_b))
            & (~candidates["object_id"].isin(forbidden))
        ]["object_id"].tolist()

        if a_candidates and b_candidates:
            for a in sorted(a_candidates):
                for b in sorted(b_candidates):
                    if a != b:
                        return a, b

    ids = sorted([oid for oid in valid_ids if oid not in forbidden])

    if len(ids) >= 2:
        return ids[0], ids[1]

    ids = sorted(valid_ids)

    if len(ids) >= 2:
        return ids[0], ids[1]

    return ids[0], None


def get_prompt_template_local(language, operator):
    """Return the prompt template used by the demo."""

    language = str(language).lower()
    operator = str(operator).lower()

    english_templates = {
        "distance": """You are given a top-down view of an indoor 3D scene and a numbered object list.\n\nObject list:\n{object_list}\n\nQuestion:\nWhat is the distance between {label_a} and {label_b}?\n\nReturn only valid JSON in this format:\n{{\"object_a\": \"{example_a}\", \"object_b\": \"{example_b}\"}}""",
        "nearest": """You are given a top-down view of an indoor 3D scene and a numbered object list.\n\nObject list:\n{object_list}\n\nQuestion:\nWhich object is nearest to {label_a}?\n\nReturn only valid JSON in this format:\n{{\"object_a\": \"{example_a}\", \"object_b\": \"{example_b}\"}}""",
    }

    portuguese_templates = {
        "distance": """Você receberá uma vista superior de uma cena 3D interna e uma lista numerada de objetos.\n\nLista de objetos:\n{object_list}\n\nPergunta:\nQual é a distância entre {label_a} e {label_b}?\n\nResponda apenas com um JSON válido neste formato:\n{{\"object_a\": \"{example_a}\", \"object_b\": \"{example_b}\"}}""",
        "nearest": """Você receberá uma vista superior de uma cena 3D interna e uma lista numerada de objetos.\n\nLista de objetos:\n{object_list}\n\nPergunta:\nQual objeto está mais próximo de {label_a}?\n\nResponda apenas com um JSON válido neste formato:\n{{\"object_a\": \"{example_a}\", \"object_b\": \"{example_b}\"}}""",
    }

    templates = portuguese_templates if language == "pt" else english_templates

    if operator not in templates:
        raise ValueError(f"Operator not supported in the demo: {operator}")

    return templates[operator]


def compute_result(grounded_a, grounded_b, gt_obj_a, gt_obj_b, gt_distance, scene_id):
    """Compute grounding status, surface distance, centroid distance, and errors."""

    grounding_correct = (
        (grounded_a == gt_obj_a and grounded_b == gt_obj_b)
        or (grounded_a == gt_obj_b and grounded_b == gt_obj_a)
    )

    e_surface = None
    e_centroid = None

    if grounded_a and grounded_b:
        pts_a = mod.load_points(grounded_a, scene_id)
        pts_b = mod.load_points(grounded_b, scene_id)

        if pts_a is not None and pts_b is not None:
            e_surface = mod.surface_distance(pts_a, pts_b)
            e_centroid = mod.centroid_distance(pts_a, pts_b)

    absolute_error = None
    percentage_error = None

    if e_surface is not None:
        absolute_error = abs(e_surface - gt_distance)
        if gt_distance > 0:
            percentage_error = (absolute_error / gt_distance) * 100

    return {
        "grounding_correct": grounding_correct,
        "surface_distance": e_surface,
        "centroid_distance": e_centroid,
        "absolute_error": absolute_error,
        "percentage_error": percentage_error,
    }


def show_compact_result(result, grounded_a, grounded_b, gt_obj_a, gt_obj_b, gt_distance):
    """Render result diagnostics in a compact single-panel layout."""

    if result["grounding_correct"]:
        st.success("Grounding correct")
    else:
        st.error("Grounding error")

    m1, m2 = st.columns(2)
    with m1:
        st.metric("Reference", f"{gt_distance:.4f} m")
    with m2:
        if result["surface_distance"] is not None:
            st.metric("Surface", f"{result['surface_distance']:.4f} m")
        else:
            st.metric("Surface", "N/A")

    m3, m4 = st.columns(2)
    with m3:
        if result["centroid_distance"] is not None:
            st.metric("Centroid", f"{result['centroid_distance']:.4f} m")
        else:
            st.metric("Centroid", "N/A")
    with m4:
        if result["absolute_error"] is not None:
            st.metric("Abs. error", f"{result['absolute_error']:.4f} m")
        else:
            st.metric("Abs. error", "N/A")

    with st.expander("Object-pair diagnosis", expanded=True):
        st.code(
            f"""Expected A: {gt_obj_a}\nExpected B: {gt_obj_b}\nSelected A: {grounded_a}\nSelected B: {grounded_b}""",
            language="text",
        )


# ============================================================
# Page setup and compact CSS
# ============================================================

st.set_page_config(page_title="VLM Grounding Demo", layout="wide")

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 0.6rem !important;
        padding-bottom: 0.3rem !important;
        padding-left: 1.1rem !important;
        padding-right: 1.1rem !important;
        max-width: 100% !important;
    }
    h1 {
        font-size: 1.45rem !important;
        margin-bottom: 0.15rem !important;
    }
    h2, h3 {
        font-size: 1.0rem !important;
        margin-top: 0.25rem !important;
        margin-bottom: 0.25rem !important;
    }
    div[data-testid="stVerticalBlock"] {
        gap: 0.35rem !important;
    }
    div[data-testid="stHorizontalBlock"] {
        gap: 0.55rem !important;
    }
    div[data-testid="stMetric"] {
        background: rgba(240, 242, 246, 0.55);
        border-radius: 0.45rem;
        padding: 0.25rem 0.45rem;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.75rem !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.0rem !important;
    }
    .stTextArea textarea {
        font-size: 0.74rem !important;
        line-height: 1.1rem !important;
    }
    .stCodeBlock pre {
        font-size: 0.72rem !important;
        line-height: 1.05rem !important;
    }
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Grounding and Distance Demo for 3D Scenes")
st.caption("Single-screen layout: scene, prompt, execution, and diagnostics.")


# ============================================================
# Load data
# ============================================================

@st.cache_data(show_spinner=False)
def load_tables():
    gt = pd.read_csv(mod.GT_CSV)
    queries = pd.read_csv(mod.QUERIES_CSV)
    manifest = pd.read_csv(mod.MANIFEST_CSV)
    return gt, queries, manifest


gt_df, queries_df, manifest_df = load_tables()
queries_df = queries_df[queries_df["review_keep"] == "yes"].copy()

extra_cols = [
    "scene_id",
    "operator",
    "structured_query",
    "label_a",
    "label_b",
    "reference_object",
    "reference_label",
    "target_category",
    "answer_object",
]

for c in extra_cols:
    if c not in queries_df.columns:
        queries_df[c] = None

merged = gt_df.merge(
    queries_df[extra_cols].drop_duplicates("structured_query"),
    on=["scene_id", "operator", "structured_query"],
    how="left",
)
merged = merged[merged["operator"] == "distance"].copy()


# ============================================================
# Top control strip
# ============================================================

scene_ids = sorted(merged["scene_id"].dropna().unique().tolist())

r1c1, r1c2, r1c3, r1c4 = st.columns([1.0, 2.35, 1.05, 1.05])

with r1c1:
    scene_id = st.selectbox("Scene", scene_ids, label_visibility="collapsed")

scene_queries = merged[merged["scene_id"] == scene_id].copy()
query_options = [f"{r['query_id']} | {r['structured_query']}" for _, r in scene_queries.iterrows()]

with r1c2:
    selected_query_text = st.selectbox("Query", query_options, label_visibility="collapsed")

with r1c3:
    execution_mode = st.selectbox(
        "Mode",
        ["VLM via API", "Manual", "Oracle"],
        label_visibility="collapsed",
    )

with r1c4:
    prompt_mode = st.selectbox(
        "Prompt",
        ["original", "context"],
        label_visibility="collapsed",
    )

r2c1, r2c2, r2c3, r2c4, r2c5 = st.columns([1.0, 1.55, 1.05, 1.05, 2.0])

with r2c1:
    provider = st.selectbox("Provider", ["openai", "openrouter"], label_visibility="collapsed")

with r2c2:
    model = st.selectbox(
        "Model",
        OPENAI_MODELS if provider == "openai" else OPENROUTER_MODELS,
        label_visibility="collapsed",
    )

with r2c3:
    language = st.selectbox("Prompt language", ["en", "pt"], index=0, label_visibility="collapsed")

with r2c4:
    if prompt_mode == "context":
        sci_mode = st.selectbox(
            "SCI",
            ["L1", "L2", "L3", "L2-Ref"],
            index=2,
            label_visibility="collapsed",
        )

        if sci_mode == "L1":
            descriptor_level = 1
        elif sci_mode == "L2":
            descriptor_level = 2
        elif sci_mode == "L3":
            descriptor_level = 3
        elif sci_mode == "L2-Ref":
            descriptor_level = 2
    else:
        descriptor_level = None
        st.caption("SCI: off")

with r2c5:
    api_key = st.text_input(
        "API key",
        type="password",
        placeholder="API key, only needed for VLM mode",
        label_visibility="collapsed",
    )

selected_query_id = selected_query_text.split(" | ")[0]
row = scene_queries[scene_queries["query_id"] == selected_query_id].iloc[0]


# ============================================================
# Build scene, prompt, and ground truth
# ============================================================

sdf = manifest_df[
    (manifest_df["scene_id"] == scene_id)
    & (manifest_df["is_valid_object"] == True)
].copy()

num_to_id, id_to_num, cat_color = mod.build_number_map(sdf)

render_path = mod.RENDERS_DIR / f"{scene_id}_numbered.jpg"
render_path.parent.mkdir(parents=True, exist_ok=True)

if not render_path.exists():
    mod.render_scene_numbered(
        scene_id=scene_id,
        scene_df=sdf,
        number_map=num_to_id,
        category_colors=cat_color,
        out_path=render_path,
    )

operator = "distance"
label_a = str(row.get("label_a") or "")
label_b = str(row.get("label_b") or "")

gt_obj_a = str(row["gt_object_a"])
gt_obj_b = str(row["gt_object_b"])
gt_distance = float(row["gt_distance_m"])

if prompt_mode == "original":
    obj_list = mod.format_object_list(num_to_id, sdf)
else:
    excluded = mod.excluded_for_query(
        operator,
        label_a=label_a,
        label_b=label_b,
        reference_label=None,
        target_category=None,
    )

    scope_category = [label_a, label_b] if sci_mode == "L2-Ref" else None

    obj_list = mod.format_object_list_with_context(
        num_to_id,
        sdf,
        level=descriptor_level,
        excluded_categories=excluded,
        language=language,
        scope_category=scope_category,
    )

valid_ids = set(sdf["object_id"].astype(str).tolist())
object_ids = sorted(valid_ids)

example_a, example_b = pick_example_ids_local(
    sdf,
    "distance",
    gt_object_a=gt_obj_a,
    gt_object_b=gt_obj_b,
    label_a=label_a,
    label_b=label_b,
    seed_key=str(row["query_id"]),
)

if example_a is None or example_b is None:
    ids = sorted(valid_ids)
    example_a = ids[0] if ids else "OBJECT_A"
    example_b = ids[1] if len(ids) > 1 else "OBJECT_B"

prompt = get_prompt_template_local(language, "distance").format(
    object_list=obj_list,
    label_a=label_a,
    label_b=label_b,
    example_a=example_a,
    example_b=example_b,
)


# ============================================================
# Main single-screen layout
# ============================================================

def _safe_file_token(value):
    return (
        str(value)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )


def _result_distance_value(result):
    if not isinstance(result, dict):
        return None

    for key in [
        "e_surface",
        "surface_distance",
        "computed_surface_distance",
        "pred_distance_m",
        "distance_m",
    ]:
        value = result.get(key)
        if value is not None:
            try:
                return float(value)
            except Exception:
                pass

    return None


grounded_a = None
grounded_b = None
response = None
result = None
run = False

# ------------------------------------------------------------
# Execution controls before visual rendering
# ------------------------------------------------------------

exec_box = st.container()

with exec_box:
    if execution_mode == "VLM via API":
        c_run, c_note = st.columns([0.9, 3.0])

        with c_run:
            run = st.button("Run VLM", width="stretch")

        with c_note:
            st.caption("The VLM selects the object instances. The geometric engine computes the distance.")

        if run:
            if not api_key:
                st.error("Enter the API key before calling the VLM.")
                st.stop()

            with st.spinner(f"Calling {model}..."):
                response = mod.call_with_retry(
                    provider=provider,
                    api_url=API_ENDPOINTS_LOCAL[provider],
                    api_key=api_key.strip(),
                    model=model,
                    prompt=prompt,
                    img_path=render_path,
                    temperature=0.0,
                    max_tokens=2048,
                )

            ids = mod.extract_ids(response, valid_ids, 2)
            grounded_a, grounded_b = ids[0], ids[1]

    elif execution_mode == "Manual":
        c_a, c_b, c_run = st.columns([2.2, 2.2, 0.9])

        with c_a:
            grounded_a = st.selectbox("Selected A", object_ids, label_visibility="visible")

        with c_b:
            grounded_b = st.selectbox("Selected B", object_ids, label_visibility="visible")

        with c_run:
            st.write("")
            run = st.button("Compute", width="stretch")

    else:
        c_info, c_run = st.columns([3.0, 0.9])

        with c_info:
            st.caption("Oracle mode uses the benchmark ground-truth pair.")

        grounded_a = gt_obj_a
        grounded_b = gt_obj_b

        with c_run:
            run = st.button("Compute oracle", width="stretch")


# ------------------------------------------------------------
# Compute result when requested
# ------------------------------------------------------------

should_compute = bool(grounded_a and grounded_b and (execution_mode == "VLM via API" or run))

if should_compute:
    result = compute_result(
        grounded_a=grounded_a,
        grounded_b=grounded_b,
        gt_obj_a=gt_obj_a,
        gt_obj_b=gt_obj_b,
        gt_distance=gt_distance,
        scene_id=scene_id,
    )

distance_for_overlay = _result_distance_value(result)

# ------------------------------------------------------------
# Build visual render paths
# ------------------------------------------------------------

demo_visual_dir = ROOT / "artifacts" / "demo_visuals"
demo_visual_dir.mkdir(parents=True, exist_ok=True)

query_token = _safe_file_token(row["query_id"])

topdown_display_path = render_path

perspective_base_path = find_perspective_image(ROOT, scene_id)
perspective_display_path = perspective_base_path

# Fallback if no manually provided perspective image exists.
scene3d_fallback_path = demo_visual_dir / f"{scene_id}_{query_token}_3d_fallback.png"

if grounded_a and grounded_b:
    a_token = _safe_file_token(grounded_a)
    b_token = _safe_file_token(grounded_b)

    topdown_display_path = demo_visual_dir / f"{scene_id}_{query_token}_{a_token}_{b_token}_topdown.jpg"

    try:
        draw_topdown_overlay(
            render_path=render_path,
            scene_df=sdf,
            selected_a=grounded_a,
            selected_b=grounded_b,
            distance_m=distance_for_overlay,
            out_path=topdown_display_path,
            scene_id=scene_id,
            load_points=mod.load_points,
        )
    except Exception as exc:
        topdown_display_path = render_path
        st.warning(f"Could not draw top-down overlay: {exc}")

    if perspective_base_path is not None:
        perspective_display_path = demo_visual_dir / f"{scene_id}_{query_token}_{a_token}_{b_token}_perspective.png"

        try:
            draw_perspective_overlay(
                image_path=perspective_base_path,
                root=ROOT,
                scene_id=scene_id,
                selected_a=grounded_a,
                selected_b=grounded_b,
                distance_m=distance_for_overlay,
                out_path=perspective_display_path,
            )
        except Exception as exc:
            perspective_display_path = perspective_base_path
            st.warning(f"Could not draw perspective overlay: {exc}")

    else:
        perspective_display_path = demo_visual_dir / f"{scene_id}_{query_token}_{a_token}_{b_token}_3d_fallback.png"

        try:
            render_scene_3d_static(
                scene_id=scene_id,
                scene_df=sdf,
                load_points=mod.load_points,
                selected_a=grounded_a,
                selected_b=grounded_b,
                distance_m=distance_for_overlay,
                out_path=perspective_display_path,
            )
        except Exception as exc:
            perspective_display_path = None
            st.warning(f"Could not render 3D fallback: {exc}")

else:
    if perspective_base_path is not None:
        perspective_display_path = perspective_base_path
    else:
        try:
            if not scene3d_fallback_path.exists():
                render_scene_3d_static(
                    scene_id=scene_id,
                    scene_df=sdf,
                    load_points=mod.load_points,
                    selected_a=None,
                    selected_b=None,
                    distance_m=None,
                    out_path=scene3d_fallback_path,
                )
            perspective_display_path = scene3d_fallback_path
        except Exception as exc:
            perspective_display_path = None
            st.warning(f"Could not render 3D fallback: {exc}")


# ------------------------------------------------------------
# Visual dashboard
# ------------------------------------------------------------

left, middle, right = st.columns([1.15, 1.15, 0.90])

with left:
    st.caption("Top-down view")
    st.image(str(topdown_display_path), caption=f"{scene_id} | {row['query_id']}", width="stretch")

with middle:
    st.caption("Perspective 3D scene")

    if perspective_display_path is not None and Path(perspective_display_path).exists():
        st.image(str(perspective_display_path), caption="Perspective view", width="stretch")
    else:
        st.info("Perspective scene image unavailable.")

with right:
    st.caption("Query and result")

    st.code(
        f"""Query: {row['structured_query']}
GT A: {gt_obj_a}
GT B: {gt_obj_b}
GT distance: {gt_distance:.4f} m""",
        language="text",
    )

    if grounded_a and grounded_b:
        st.code(
            f"""Selected A: {grounded_a}
Selected B: {grounded_b}""",
            language="text",
        )

    if result is not None:
        show_compact_result(result, grounded_a, grounded_b, gt_obj_a, gt_obj_b, gt_distance)
    else:
        st.caption("Select objects or run the VLM to update the highlighted views.")

    with st.expander("Prompt", expanded=False):
        st.text_area("Prompt", prompt, height=260, label_visibility="collapsed")

    if response is not None:
        with st.expander("Raw VLM response", expanded=False):
            st.code(response, language="text")
