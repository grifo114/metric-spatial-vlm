import json
import re


# ============================================================
# VOCABULÁRIO CONTROLADO
# ============================================================

PT_TO_EN = {
    "cadeira": "chair",
    "mesa": "table",
    "escrivaninha": "desk",
    "porta": "door",
    "monitor": "monitor",
    "janela": "window",
    "armario": "cabinet",
    "arquivo": "file cabinet",
    "lixeira": "trash can",
    "planta": "plant",
    "telefone": "telephone",
    "radiador": "radiator",
    "chao": "floor",
    "piso": "floor",
    "parede": "wall",
}

VALID_LABELS = {
    "chair",
    "table",
    "desk",
    "door",
    "monitor",
    "window",
    "cabinet",
    "file cabinet",
    "trash can",
    "plant",
    "telephone",
    "radiator",
    "floor",
    "wall",
}


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def normalize_text(text: str) -> str:
    text = text.lower().strip()

    replacements = {
        "á": "a",
        "à": "a",
        "ã": "a",
        "â": "a",
        "é": "e",
        "ê": "e",
        "í": "i",
        "ó": "o",
        "ô": "o",
        "õ": "o",
        "ú": "u",
        "ç": "c",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_intent(text: str) -> str:
    distance_patterns = [
        "distance between",
        "how far",
        "qual a distancia",
        "a qual distancia",
        "distancia entre",
        "distancia da",
        "distancia do",
    ]

    for pattern in distance_patterns:
        if pattern in text:
            return "distance"

    return "unknown"


def map_token_to_label(token: str):
    if token in VALID_LABELS:
        return token
    if token in PT_TO_EN:
        return PT_TO_EN[token]
    return None


def extract_candidate_labels(text: str):
    found = []

    # primeiro procura labels compostos
    compound_labels = sorted(
        [label for label in VALID_LABELS if " " in label],
        key=len,
        reverse=True
    )
    for label in compound_labels:
        if label in text:
            found.append(label)

    # depois procura tokens simples
    for token in text.split():
        label = map_token_to_label(token)
        if label is not None and label not in found:
            found.append(label)

    return found


def parse_query(query: str):
    normalized = normalize_text(query)
    intent = detect_intent(normalized)
    candidates = extract_candidate_labels(normalized)

    result = {
        "raw_query": query,
        "normalized_query": normalized,
        "intent": intent,
        "object_a_text": None,
        "object_b_text": None,
        "candidate_labels": candidates,
    }

    if intent == "distance" and len(candidates) >= 2:
        result["object_a_text"] = candidates[0]
        result["object_b_text"] = candidates[1]

    return result


# ============================================================
# DEMO
# ============================================================

def main():
    print("\n[INFO] Natural query parser\n")
    query = input("Enter a query: ").strip()

    parsed = parse_query(query)

    print("\n[INFO] Parsed query:")
    print(json.dumps(parsed, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()