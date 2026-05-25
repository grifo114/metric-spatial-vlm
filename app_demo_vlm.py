import os
import sys
import importlib.util
from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# Configuração básica
# ============================================================

ROOT = Path(__file__).resolve().parent
SCRIPT_PATH = ROOT / "scripts" / "83_e2e_grounding_test_official_v2.py"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================
# Importa o script principal, mesmo ele começando com número
# ============================================================

spec = importlib.util.spec_from_file_location("e2e_script", SCRIPT_PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


# ============================================================
# Funções auxiliares da interface
# ============================================================

def evaluate_and_show_result(
    grounded_a,
    grounded_b,
    gt_obj_a,
    gt_obj_b,
    gt_distance,
    scene_id,
    mod,
):    

    def pick_example_ids_local(
        sdf,
        operator,
        gt_object_a=None,
        gt_object_b=None,
        label_a=None,
        label_b=None,
        seed_key=None,
    ):
        """
        Escolhe dois objetos válidos para usar como exemplo no prompt.
        Evita usar o próprio par de ground truth quando possível.
        """

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

        # O manifesto do seu projeto usa label_norm, não category
        if "label_norm" in candidates.columns:
            a_candidates = candidates[
                (candidates["label_norm"].astype(str) == str(label_a)) &
                (~candidates["object_id"].isin(forbidden))
            ]["object_id"].tolist()

            b_candidates = candidates[
                (candidates["label_norm"].astype(str) == str(label_b)) &
                (~candidates["object_id"].isin(forbidden))
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
        return ids[0], ids[1]


    def get_prompt_template_local(language, operator):
        """
        Retorna o template de prompt usado pela demonstração.
        Por enquanto, o script 83 só tem prompt em português.
        """

        if operator == "distance":
            return mod.PROMPT_DISTANCE

        if operator == "nearest":
            return mod.PROMPT_NEAREST

        raise ValueError(f"Operador não suportado no demo: {operator}")


    API_ENDPOINTS_LOCAL = {
        "openai": "https://api.openai.com/v1/chat/completions",
        "openrouter": "https://openrouter.ai/api/v1/chat/completions",
    }


    """
    Avalia o par selecionado, calcula a distância e mostra o diagnóstico.
    Esta função é usada nos três modos:
    - VLM via API
    - Manual sem VLM
    - Oracle ground truth
    """

    grounding_correct = (
        (grounded_a == gt_obj_a and grounded_b == gt_obj_b) or
        (grounded_a == gt_obj_b and grounded_b == gt_obj_a)
    )

    # ============================================================
    # Diagnóstico específico do grounding
    # ============================================================

    st.subheader("Diagnóstico do grounding")

    a_correct = grounded_a == gt_obj_a
    b_correct = grounded_b == gt_obj_b

    if grounding_correct:
        st.success(
            "O grounding está correto: as instâncias selecionadas correspondem "
            "às instâncias esperadas no benchmark."
        )
    else:
        st.error(
            "Erro de grounding: pelo menos uma instância selecionada é diferente "
            "da instância esperada no benchmark."
        )

        if a_correct:
            st.success(f"Objeto A correto: {grounded_a}")
        else:
            st.warning(
                f"Objeto A incorreto: esperado {gt_obj_a}, "
                f"selecionado {grounded_a}"
            )

        if b_correct:
            st.success(f"Objeto B correto: {grounded_b}")
        else:
            st.warning(
                f"Objeto B incorreto: esperado {gt_obj_b}, "
                f"selecionado {grounded_b}"
            )

    st.subheader("Objetos usados no cálculo")

    st.code(
        f"""
Objeto A esperado:      {gt_obj_a}
Objeto B esperado:      {gt_obj_b}

Objeto A selecionado:   {grounded_a}
Objeto B selecionado:   {grounded_b}

Grounding correto:      {grounding_correct}
""",
        language="text",
    )

    # ============================================================
    # Cálculo geométrico
    # ============================================================

    e_surface = None
    e_centroid = None

    if grounded_a and grounded_b:
        pts_a = mod.load_points(grounded_a, scene_id)
        pts_b = mod.load_points(grounded_b, scene_id)

        if pts_a is not None and pts_b is not None:
            e_surface = mod.surface_distance(pts_a, pts_b)
            e_centroid = mod.centroid_distance(pts_a, pts_b)

    st.subheader("Diagnóstico do resultado")

    if e_surface is not None:
        surface_error_m = abs(e_surface - gt_distance)

        if gt_distance > 0:
            surface_error_pct = (surface_error_m / gt_distance) * 100
        else:
            surface_error_pct = None

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.metric("Distância correta", f"{gt_distance:.4f} m")

        with col_b:
            st.metric("Distância calculada", f"{e_surface:.4f} m")

        with col_c:
            st.metric("Erro absoluto", f"{surface_error_m:.4f} m")

        if surface_error_pct is not None:
            st.metric("Erro percentual", f"{surface_error_pct:.2f}%")
        else:
            st.warning(
                "Não foi possível calcular erro percentual porque a distância correta é zero."
            )

        st.subheader("Quem errou?")

        if not grounding_correct:
            st.error(
                "O erro veio da escolha das instâncias. A engine geométrica "
                "calculou a distância entre os objetos selecionados, mas esse "
                "par não corresponde ao par correto do benchmark."
            )
        else:
            st.success(
                "As instâncias foram selecionadas corretamente. Neste caso, "
                "a distância foi calculada entre os objetos corretos do benchmark."
            )

        st.subheader("Comparação das formas de distância")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Surface distance", f"{e_surface:.4f} m")

        with col2:
            st.metric("Centroid distance", f"{e_centroid:.4f} m")

        with col3:
            st.metric("Ground truth distance", f"{gt_distance:.4f} m")

    else:
        st.warning(
            "Não foi possível calcular a distância. Pontos dos objetos não encontrados."
        )

def pick_example_ids_local(
    sdf,
    operator,
    gt_object_a=None,
    gt_object_b=None,
    label_a=None,
    label_b=None,
    seed_key=None,
):
    """
    Escolhe dois objetos válidos para usar como exemplo no prompt.
    Evita usar o próprio par de ground truth quando possível.
    """

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
            (candidates["label_norm"].astype(str) == str(label_a)) &
            (~candidates["object_id"].isin(forbidden))
        ]["object_id"].tolist()

        b_candidates = candidates[
            (candidates["label_norm"].astype(str) == str(label_b)) &
            (~candidates["object_id"].isin(forbidden))
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
    """
    Retorna o template de prompt usado pela demo.
    No script 83 os prompts disponíveis são PROMPT_DISTANCE e PROMPT_NEAREST.
    """

    if operator == "distance":
        return mod.PROMPT_DISTANCE

    if operator == "nearest":
        return mod.PROMPT_NEAREST

    raise ValueError(f"Operador não suportado: {operator}")


API_ENDPOINTS_LOCAL = {
    "openai": "https://api.openai.com/v1/chat/completions",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
}
# ============================================================
# Interface
# ============================================================

st.set_page_config(page_title="Demo VLM Grounding", layout="wide")

st.title("Sistema Demonstrativo de Grounding e Distância em Cenas 3D")

st.caption(
    "Seleção de cena, consulta espacial, escolha de instâncias e cálculo geométrico "
    "de distância. O sistema permite testar com VLM, seleção manual ou ground truth."
)


# ============================================================
# Carrega os dados
# ============================================================

gt_df = pd.read_csv(mod.GT_CSV)
queries_df = pd.read_csv(mod.QUERIES_CSV)
manifest_df = pd.read_csv(mod.MANIFEST_CSV)

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

# Para a demonstração, vamos focar em distance
merged = merged[merged["operator"] == "distance"].copy()


# ============================================================
# Seleção da cena e da query
# ============================================================

scene_ids = sorted(merged["scene_id"].dropna().unique().tolist())

col1, col2 = st.columns([1, 2])

with col1:
    scene_id = st.selectbox("Escolha a cena", scene_ids)

scene_queries = merged[merged["scene_id"] == scene_id].copy()

query_options = []
for _, r in scene_queries.iterrows():
    query_options.append(
        f"{r['query_id']} | {r['structured_query']}"
    )

with col2:
    selected_query_text = st.selectbox("Escolha a query", query_options)

selected_query_id = selected_query_text.split(" | ")[0]
row = scene_queries[scene_queries["query_id"] == selected_query_id].iloc[0]


# ============================================================
# Configurações do experimento
# ============================================================

st.subheader("Configuração do experimento")

execution_mode = st.selectbox(
    "Modo de execução",
    ["VLM via API", "Manual sem VLM", "Oracle ground truth"],
    help=(
        "VLM via API chama o modelo. Manual sem VLM permite selecionar os objetos "
        "manualmente. Oracle ground truth usa diretamente o par correto do benchmark."
    ),
)

c1, c2, c3, c4 = st.columns(4)

with c1:
    provider = st.selectbox(
        "Provider",
        ["openai", "openrouter"],
        help="OpenAI usa modelos GPT. OpenRouter permite usar modelos Qwen3-VL."
    )

OPENAI_MODELS = [
    "gpt-4.1",
    "gpt-4.1-mini",
]

OPENROUTER_MODELS = [
    "qwen/qwen3-vl-8b-instruct",
    "qwen/qwen3-vl-32b-instruct",
    "qwen/qwen3-vl-235b-a22b-instruct",
]

with c2:
    if provider == "openai":
        model = st.selectbox("VLM", OPENAI_MODELS)
    else:
        model = st.selectbox("VLM", OPENROUTER_MODELS)

with c3:
    prompt_mode = st.selectbox("Prompt mode", ["original", "context"])

with c4:
    language = st.selectbox("Idioma", ["en", "pt"], index=0)

descriptor_level = None
if prompt_mode == "context":
    descriptor_level = st.selectbox("Nível SCI", [1, 2, 3], index=1)

api_key = st.text_input(
    "API key",
    type="password",
    placeholder="Cole sua chave aqui. Ela não ficará visível na tela.",
    help="A chave é usada apenas durante esta execução local do Streamlit."
)


# ============================================================
# Carrega objetos da cena e renderiza imagem
# ============================================================

sdf = manifest_df[
    (manifest_df["scene_id"] == scene_id) &
    (manifest_df["is_valid_object"] == True)
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


# ============================================================
# Monta a lista de objetos
# ============================================================

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

    obj_list = mod.format_object_list_with_context(
        num_to_id,
        sdf,
        level=descriptor_level,
        excluded_categories=excluded,
        language=language,
        scope_category=None,
    )


# ============================================================
# Escolhe exemplo adaptativo e monta prompt
# ============================================================

valid_ids = set(sdf["object_id"].tolist())

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
# Exibição visual
# ============================================================

left, right = st.columns([1, 1])

with left:
    st.subheader("Imagem enviada ao VLM")
    st.image(str(render_path), caption=f"Cena: {scene_id}", use_container_width=True)

with right:
    st.subheader("Prompt enviado ao VLM")
    st.text_area("Prompt", prompt, height=520)


# ============================================================
# Informações do ground truth
# ============================================================

st.subheader("Ground truth da query")

st.code(
    f"""
Query ID: {row['query_id']}
Cena: {scene_id}
Consulta: {row['structured_query']}

Objeto A esperado: {gt_obj_a}
Objeto B esperado: {gt_obj_b}
Distância de referência: {gt_distance:.4f} m
""",
    language="text",
)


# ============================================================
# Execução
# ============================================================

st.subheader("Execução")

object_ids = sorted(valid_ids)

# ------------------------------------------------------------
# Modo 1: VLM via API
# ------------------------------------------------------------

if execution_mode == "VLM via API":
    run_api = st.button("Enviar imagem + prompt para o VLM")

    if run_api:
        api_url = API_ENDPOINTS_LOCAL[provider]

        if not api_key:
            st.error("Informe a API key no campo acima antes de chamar o VLM.")
            st.stop()

        api_key = api_key.strip()

        with st.spinner(f"Chamando o VLM: {model}..."):
            response = mod.call_with_retry(
                provider=provider,
                api_url=api_url,
                api_key=api_key,
                model=model,
                prompt=prompt,
                img_path=render_path,
                temperature=0.0,
                max_tokens=2048,
            )

        st.subheader("Resposta bruta do VLM")
        st.code(response, language="text")

        ids = mod.extract_ids(response, valid_ids, 2)
        grounded_a, grounded_b = ids[0], ids[1]

        st.subheader("Objetos selecionados pelo VLM")

        st.code(
            f"""
Objeto A selecionado: {grounded_a}
Objeto B selecionado: {grounded_b}
""",
            language="text",
        )

        evaluate_and_show_result(
            grounded_a=grounded_a,
            grounded_b=grounded_b,
            gt_obj_a=gt_obj_a,
            gt_obj_b=gt_obj_b,
            gt_distance=gt_distance,
            scene_id=scene_id,
            mod=mod,
        )


# ------------------------------------------------------------
# Modo 2: Manual sem VLM
# ------------------------------------------------------------

elif execution_mode == "Manual sem VLM":
    st.info(
        "Neste modo, você simula manualmente o papel do VLM. "
        "Escolha o Objeto A e o Objeto B, e o sistema calculará a distância."
    )

    col_a, col_b = st.columns(2)

    with col_a:
        grounded_a = st.selectbox("Selecione manualmente o Objeto A", object_ids)

    with col_b:
        grounded_b = st.selectbox("Selecione manualmente o Objeto B", object_ids)

    run_manual = st.button("Calcular distância sem VLM")

    if run_manual:
        evaluate_and_show_result(
            grounded_a=grounded_a,
            grounded_b=grounded_b,
            gt_obj_a=gt_obj_a,
            gt_obj_b=gt_obj_b,
            gt_distance=gt_distance,
            scene_id=scene_id,
            mod=mod,
        )


# ------------------------------------------------------------
# Modo 3: Oracle ground truth
# ------------------------------------------------------------

elif execution_mode == "Oracle ground truth":
    st.info(
        "Neste modo, o sistema usa diretamente o par correto do benchmark. "
        "Isso mostra o comportamento da engine geométrica quando o grounding está correto."
    )

    st.code(
        f"""
Objeto A usado: {gt_obj_a}
Objeto B usado: {gt_obj_b}
""",
        language="text",
    )

    run_oracle = st.button("Calcular usando ground truth")

    if run_oracle:
        evaluate_and_show_result(
            grounded_a=gt_obj_a,
            grounded_b=gt_obj_b,
            gt_obj_a=gt_obj_a,
            gt_obj_b=gt_obj_b,
            gt_distance=gt_distance,
            scene_id=scene_id,
            mod=mod,
        )