"""
query_parser.py
Parser de linguagem natural para operadores espaciais.
Suporte bilíngue: PT-BR e EN.

Operadores:
    distance(obj_a, obj_b)         → distância entre dois objetos
    nearest(ref_obj, category)     → objeto mais próximo de uma referência
    between(obj_x, obj_a, obj_b)   → obj_x está entre obj_a e obj_b?
    aligned(obj_a, obj_b, obj_c)   → três objetos estão alinhados?
"""

import re
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Resultado do parsing
# ---------------------------------------------------------------------------

@dataclass
class ParsedQuery:
    operator: str                   # distance | nearest | between | aligned
    entities: list[str]             # nomes extraídos da query
    category: Optional[str] = None  # para nearest: categoria alvo
    confidence: float = 1.0
    raw: str = ""

    def to_dict(self):
        return {
            "operator":   self.operator,
            "entities":   self.entities,
            "category":   self.category,
            "confidence": self.confidence,
            "raw":        self.raw,
        }


# ---------------------------------------------------------------------------
# Vocabulário bilíngue
# ---------------------------------------------------------------------------

# Padrões de DISTÂNCIA
DISTANCE_PATTERNS = [
    # PT-BR
    r"(?:qual|quanto|me diga|calcule?|informe?).*dist[âa]ncia.*entre\s+(.+?)\s+e\s+(.+?)[\?\.]?$",
    r"dist[âa]ncia\s+(?:entre\s+)?(.+?)\s+e\s+(.+?)[\?\.]?$",
    r"qu[aã]o\s+(?:longe|perto|distante).*(.+?)\s+(?:est[áa]|fica).*(?:de|do|da)\s+(.+?)[\?\.]?$",
    # EN
    r"(?:what\s+is\s+the\s+)?distance\s+between\s+(.+?)\s+and\s+(.+?)[\?\.]?$",
    r"how\s+(?:far|close)\s+(?:is\s+)?(.+?)\s+(?:from|to)\s+(.+?)[\?\.]?$",
]

# Padrões de NEAREST
NEAREST_PATTERNS = [
    # PT-BR
    r"qual\s+(.+?)\s+(?:est[áa]|fica)\s+mais\s+pr[óo]xim[oa]\s+(?:de|do|da|ao|à)\s+(.+?)[\?\.]?$",
    r"qual\s+(?:é\s+a\s+)?(.+?)\s+mais\s+pr[óo]xim[oa]\s+(?:de|do|da|ao|à)\s+(.+?)[\?\.]?$",
    r"(?:me\s+diga\s+)?qual\s+(.+?)\s+(?:se\s+encontra|está)\s+mais\s+perto\s+(?:de|do|da)\s+(.+?)[\?\.]?$",
    r"(.+?)\s+mais\s+pr[óo]xim[oa]\s+(?:de|do|da)\s+(.+?)[\?\.]?$",
    # EN
    r"which\s+(.+?)\s+is\s+(?:the\s+)?(?:nearest|closest)\s+(?:to\s+)?(.+?)[\?\.]?$",
    r"(?:what|which)\s+(.+?)\s+is\s+closest\s+to\s+(.+?)[\?\.]?$",
    r"find\s+(?:the\s+)?(?:nearest|closest)\s+(.+?)\s+to\s+(.+?)[\?\.]?$",
]

# Padrões de BETWEEN
BETWEEN_PATTERNS = [
    # PT-BR
    r"(.+?)\s+(?:est[áa]|fica|se\s+encontra)\s+entre\s+(?:[ao]\s+)?(.+?)\s+e\s+(?:[ao]\s+)?(.+?)[\?\.]?$",
    r"(.+?)\s+est[áa]\s+(?:no\s+meio|entre)\s+(?:[ao]\s+)?(.+?)\s+e\s+(?:[ao]\s+)?(.+?)[\?\.]?$",
    r"(?:verifique?|diga)\s+se\s+(.+?)\s+(?:est[áa]|fica)\s+entre\s+(.+?)\s+e\s+(.+?)[\?\.]?$",
    # EN
    r"(?:is\s+)?(.+?)\s+between\s+(.+?)\s+and\s+(.+?)[\?\.]?$",
    r"(?:check\s+if\s+)?(.+?)\s+(?:is\s+)?in\s+between\s+(.+?)\s+and\s+(.+?)[\?\.]?$",
]

# Padrões de ALIGNED
ALIGNED_PATTERNS = [
    # PT-BR
    r"(.+?),\s*(.+?)\s+e\s+(.+?)\s+(?:est[ãa]o|est[áa])\s+alinhad[oa]s?[\?\.]?$",
    r"(.+?),\s*(.+?)\s+e\s+(.+?)\s+(?:formam|est[ãa]o\s+em)\s+linha[\?\.]?$",
    r"(?:verifique?|diga)\s+se\s+(.+?),\s*(.+?)\s+e\s+(.+?)\s+est[ãa]o\s+alinhad[oa]s?[\?\.]?$",
    # EN
    r"(?:are\s+)?(.+?),\s*(.+?)\s+and\s+(.+?)\s+aligned[\?\.]?$",
    r"(?:is\s+there\s+)?(.+?),\s*(.+?)\s+and\s+(.+?)\s+in\s+(?:a\s+)?line[\?\.]?$",
    r"(?:check\s+if\s+)?(.+?),\s*(.+?)\s+and\s+(.+?)\s+are\s+(?:in\s+)?alignment[\?\.]?$",
]


# ---------------------------------------------------------------------------
# Mapeamento de categorias bilíngue
# ---------------------------------------------------------------------------

CATEGORY_MAP = {
    # PT-BR → nome interno
    "cadeira": "chair",    "cadeiras": "chair",
    "sofá": "sofa",        "sofas": "sofa",       "sofás": "sofa",
    "mesa": "dining_table","mesas": "dining_table",
    "escrivaninha": "desk","escrivaninhas": "desk",
    "armário": "wardrobe", "armários": "wardrobe",
    "cama": "bed",         "camas": "bed",
    "monitor": "monitor",  "monitores": "monitor",
    "porta": "door",       "portas": "door",
    # EN → nome interno
    "chair": "chair",      "chairs": "chair",
    "sofa": "sofa",        "sofas": "sofa",
    "table": "dining_table","tables": "dining_table",
    "desk": "desk",        "desks": "desk",
    "wardrobe": "wardrobe","wardrobes": "wardrobe",
    "cabinet": "wardrobe", "cabinets": "wardrobe",
    "bed": "bed",          "beds": "bed",
    "door": "door",        "doors": "door",
}


def normalize_category(text: str) -> Optional[str]:
    """Normaliza nome de categoria para o padrão interno."""
    text = text.strip().lower()
    return CATEGORY_MAP.get(text)


def clean_entity(text: str) -> str:
    """Remove artigos e preposições dos nomes de entidade."""
    text = text.strip()
    # Remove artigos PT-BR
    for art in ["o ", "a ", "os ", "as ", "um ", "uma ",
                "do ", "da ", "dos ", "das ", "ao ", "à "]:
        if text.lower().startswith(art):
            text = text[len(art):]
    # Remove artigos EN
    for art in ["the ", "a ", "an "]:
        if text.lower().startswith(art):
            text = text[len(art):]
    return text.strip()


# ---------------------------------------------------------------------------
# Funções de parsing por operador
# ---------------------------------------------------------------------------

def _try_patterns(query: str, patterns: list[str]) -> Optional[re.Match]:
    q = query.strip().lower()
    for pattern in patterns:
        m = re.search(pattern, q, re.IGNORECASE)
        if m:
            return m
    return None


def parse_distance(query: str) -> Optional[ParsedQuery]:
    m = _try_patterns(query, DISTANCE_PATTERNS)
    if not m:
        return None
    a = clean_entity(m.group(1))
    b = clean_entity(m.group(2))
    return ParsedQuery(
        operator="distance",
        entities=[a, b],
        raw=query,
    )


def parse_nearest(query: str) -> Optional[ParsedQuery]:
    m = _try_patterns(query, NEAREST_PATTERNS)
    if not m:
        return None
    # grupo 1 = categoria alvo, grupo 2 = objeto de referência
    cat_raw = clean_entity(m.group(1))
    ref     = clean_entity(m.group(2))
    cat     = normalize_category(cat_raw)
    return ParsedQuery(
        operator="nearest",
        entities=[ref],
        category=cat or cat_raw,
        raw=query,
    )


def parse_between(query: str) -> Optional[ParsedQuery]:
    m = _try_patterns(query, BETWEEN_PATTERNS)
    if not m:
        return None
    x = clean_entity(m.group(1))
    a = clean_entity(m.group(2))
    b = clean_entity(m.group(3))
    return ParsedQuery(
        operator="between",
        entities=[x, a, b],
        raw=query,
    )


def parse_aligned(query: str) -> Optional[ParsedQuery]:
    m = _try_patterns(query, ALIGNED_PATTERNS)
    if not m:
        return None
    a = clean_entity(m.group(1))
    b = clean_entity(m.group(2))
    c = clean_entity(m.group(3))
    return ParsedQuery(
        operator="aligned",
        entities=[a, b, c],
        raw=query,
    )


# ---------------------------------------------------------------------------
# Parser principal
# ---------------------------------------------------------------------------

def parse_query(query: str) -> Optional[ParsedQuery]:
    """
    Tenta parsear a query em PT-BR ou EN.
    Retorna None se não reconhecer o operador.
    """
    for fn in [parse_distance, parse_nearest, parse_between, parse_aligned]:
        result = fn(query)
        if result:
            return result
    return None


def parse_query_or_error(query: str) -> dict:
    """Retorna dict com resultado ou mensagem de erro bilíngue."""
    result = parse_query(query)
    if result:
        return {"ok": True, "parsed": result.to_dict()}
    return {
        "ok": False,
        "error": (
            "Não consegui identificar o operador espacial na sua consulta. "
            "Tente formatos como:\n"
            "• 'Qual a distância entre A e B?'\n"
            "• 'Qual cadeira está mais próxima do armário?'\n"
            "• 'A cadeira está entre a mesa e o sofá?'\n"
            "• 'A, B e C estão alinhados?'\n\n"
            "Could not identify the spatial operator. Try:\n"
            "• 'What is the distance between A and B?'\n"
            "• 'Which chair is closest to the wardrobe?'\n"
            "• 'Is the chair between the table and the sofa?'\n"
            "• 'Are A, B and C aligned?'"
        ),
    }


# ---------------------------------------------------------------------------
# Teste rápido
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    queries = [
        "Qual a distância entre a cadeira e o monitor?",
        "Qual cadeira está mais próxima do armário?",
        "A cadeira está entre o armário 1 e o armário 2?",
        "Mesa, cadeira e escrivaninha estão alinhados?",
        "What is the distance between chair_0 and table_1?",
        "Which chair is closest to the wardrobe?",
        "Is chair_2 between cabinet_0 and cabinet_1?",
        "Are chair_0, desk_0 and monitor_0 aligned?",
    ]
    for q in queries:
        r = parse_query(q)
        print(f"Q: {q}")
        print(f"   → {r}\n")
