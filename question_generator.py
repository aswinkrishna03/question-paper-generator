import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import logging as hf_logging
import random
import re
import os
from gemini_validator import generate_questions_with_gemini

MODEL_NAME = os.environ.get("QPG_MODEL_NAME", "microsoft/Phi-3-mini-4k-instruct")
BLOOMS_LEVELS = [
    "Remember",
    "Understand",
    "Apply",
    "Analyze",
    "Evaluate",
    "Create",
]
DIST_PROFILES = {
    "balanced": {
        "Remember": 15,
        "Understand": 20,
        "Apply": 20,
        "Analyze": 20,
        "Evaluate": 15,
        "Create": 10,
    },
    "foundational": {
        "Remember": 25,
        "Understand": 25,
        "Apply": 20,
        "Analyze": 15,
        "Evaluate": 10,
        "Create": 5,
    },
    "advanced": {
        "Remember": 5,
        "Understand": 10,
        "Apply": 20,
        "Analyze": 25,
        "Evaluate": 20,
        "Create": 20,
    },
}
PAPER_MODELS = {
    "50": {
        "label": "50 marks",
        "total_marks": 50,
        "sections": [
            {
                "name": "SECTION A",
                "count": 6,
                "marks": 3,
                "question_style": "SHORT",
                "instruction": "Each question must be ONE sentence under 12 words.",
            },
            {
                "name": "SECTION B",
                "count": 2,
                "marks": 16,
                "question_style": "LONG analytical",
                "instruction": "Each question must be clear, analytical, and syllabus-aligned.",
            },
        ],
    },
    "100": {
        "label": "100 marks",
        "total_marks": 100,
        "sections": [
            {
                "name": "SECTION A",
                "count": 10,
                "marks": 3,
                "question_style": "SHORT",
                "instruction": "Each question must be ONE sentence under 12 words.",
            },
            {
                "name": "SECTION B",
                "count": 5,
                "marks": 14,
                "question_style": "LONG analytical",
                "instruction": "Each question must be clear, analytical, and syllabus-aligned.",
            },
        ],
    },
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_TOKENIZER = None
_MODEL = None


def _get_model_and_tokenizer():

    global _TOKENIZER, _MODEL

    if _TOKENIZER is not None and _MODEL is not None:
        return _TOKENIZER, _MODEL

    hf_logging.set_verbosity_error()
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32

    _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
    if DEVICE == "cuda":
        _MODEL = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            dtype=dtype
        )
    else:
        _MODEL = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=dtype
        )
        _MODEL.to(DEVICE)

    return _TOKENIZER, _MODEL


def get_runtime_device():

    return DEVICE


def normalize_blooms_levels(levels):

    if not levels:
        return BLOOMS_LEVELS

    normalized = []
    allowed = {level.lower(): level for level in BLOOMS_LEVELS}

    for level in levels:
        clean = str(level).strip().lower()
        if clean in allowed and allowed[clean] not in normalized:
            normalized.append(allowed[clean])

    return normalized if normalized else BLOOMS_LEVELS


def normalize_distribution_profile(profile):

    clean = str(profile or "balanced").strip().lower()
    return clean if clean in DIST_PROFILES else "balanced"


def normalize_paper_model(model):

    clean = str(model or "50").strip()
    return clean if clean in PAPER_MODELS else "50"


def get_paper_model(model):

    return PAPER_MODELS[normalize_paper_model(model)]


def resolve_distribution(levels, profile):

    selected_levels = normalize_blooms_levels(levels)
    profile_name = normalize_distribution_profile(profile)
    base = DIST_PROFILES[profile_name]

    selected_weight_sum = sum(base[level] for level in selected_levels)
    if selected_weight_sum == 0:
        equal = 100 / len(selected_levels)
        return {level: equal for level in selected_levels}, profile_name

    adjusted = {
        level: (base[level] / selected_weight_sum) * 100 for level in selected_levels
    }
    return adjusted, profile_name


def allocate_question_counts(distribution, total_questions=7):

    exact = {
        level: (percent / 100) * total_questions
        for level, percent in distribution.items()
    }
    counts = {level: int(value) for level, value in exact.items()}
    assigned = sum(counts.values())

    remainders = sorted(
        ((level, exact[level] - counts[level]) for level in distribution),
        key=lambda item: item[1],
        reverse=True
    )

    idx = 0
    while assigned < total_questions and remainders:
        level = remainders[idx % len(remainders)][0]
        counts[level] += 1
        assigned += 1
        idx += 1

    return counts


def build_output_template(sections):

    lines = []

    for section in sections:
        lines.append(section["name"])
        for idx in range(1, section["count"] + 1):
            lines.append(f"{idx}. [Module: <1-5>] [Bloom: <Level>] <Question Text>")
        lines.append("")

    return "\n".join(lines).strip()


def build_section_instructions(sections):

    lines = []

    for section in sections:
        lines.append(section["name"])
        lines.append(
            f"Generate {section['count']} {section['question_style']} questions ({section['marks']} marks each)."
        )
        lines.append(section["instruction"])
        if section["question_style"].startswith("SHORT"):
            lines.append("Prefer lower-to-mid Bloom levels from the selected list.")
        else:
            lines.append("Prefer higher Bloom levels from the selected list.")
        lines.append("")

    return "\n".join(lines).strip()


def build_module_instructions(module_concepts, paper_model_key):

    if not module_concepts:
        return ""

    module_lines = ["Modules and extracted concepts:"]
    for module in module_concepts:
        concept_text = ", ".join(module["concepts"]) if module["concepts"] else "No strong concepts extracted"
        module_lines.append(
            f"- Module {module['module']}: {concept_text}"
        )

    if paper_model_key == "100":
        module_lines.append("")
        module_lines.append("Mandatory module-wise distribution for 100 marks model:")
        module_lines.append("- SECTION A: exactly 2 questions from each module (Modules 1 to 5).")
        module_lines.append("- SECTION B: exactly 1 question from each module (Modules 1 to 5).")
        module_lines.append("- Do not repeat module allocation beyond these counts.")
    else:
        module_lines.append("")
        module_lines.append("For 50 marks model, module order can be flexible.")
        module_lines.append("Ensure questions are covered from the provided modules.")

    module_lines.append("")
    module_lines.append("Each question must start with module and Bloom tags in this exact order:")
    module_lines.append("[Module: <1-5>] [Bloom: <Level>] <question text>")

    return "\n".join(module_lines).strip()


def build_prompt(
    concepts,
    blooms_levels=None,
    blooms_distribution="balanced",
    paper_model_key="50",
    module_concepts=None
):

    topics = ", ".join(concepts)
    selected_levels = normalize_blooms_levels(blooms_levels)
    levels_text = ", ".join(selected_levels)
    paper_model = get_paper_model(paper_model_key)
    sections = paper_model["sections"]
    total_questions = sum(section["count"] for section in sections)
    distribution, profile_name = resolve_distribution(
        selected_levels, blooms_distribution
    )
    counts = allocate_question_counts(distribution, total_questions=total_questions)
    distribution_text = "\n".join(
        [
            f"- {level}: {distribution[level]:.1f}% (target {counts[level]} of {total_questions} questions)"
            for level in selected_levels
        ]
    )
    section_text = build_section_instructions(sections)
    module_text = build_module_instructions(module_concepts or [], paper_model_key)
    output_template = build_output_template(sections)

    prompt = f"""
You are a university professor.

Topics:
{topics}

Bloom's taxonomy levels to use:
{levels_text}

Distribution profile:
{profile_name}

Question paper model:
{paper_model["label"]} (Total Marks: {paper_model["total_marks"]})

Target Bloom distribution:
{distribution_text}

Create an exam question paper.

{section_text}

{module_text}

For each question, add tags in this exact format:
[Module: <1-5>] [Bloom: <Level>] <question text>
Use only these levels:
{levels_text}
Match the target distribution as closely as possible across all {total_questions} questions.

Output strictly in this format:

{output_template}
"""

    return prompt


def generate_paper(prompt):

    tokenizer, model = _get_model_and_tokenizer()
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_len = inputs["input_ids"].shape[1]

    max_tokens = 180 if DEVICE == "cpu" else 350
    outputs = model.generate(
        **inputs,
        do_sample=True,
        max_new_tokens=max_tokens,
        temperature=0.6,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

    generated_ids = outputs[0][input_len:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return text


def generate_paper_gemini(prompt):

    result = generate_questions_with_gemini(prompt)
    if not result.get("ok"):
        return ""
    return str(result.get("text", "")).strip()


def _normalize_concept(concept):

    text = re.sub(r"\s+", " ", str(concept or "").strip())
    text = text.strip(" .,:;\"'")
    return text


def _sanitize_topic(topic):

    clean = _normalize_concept(topic)
    clean = re.sub(r"\s*[-–—]\s*", " - ", clean)
    clean = re.sub(r"^[A-Z][A-Z\s]{3,}\s*-\s*", "", clean)
    clean = re.sub(r"\b([A-Za-z]+)\s+s\b", r"\1's", clean)
    clean = re.sub(r"([a-z])([A-Z])", r"\1 \2", clean)
    clean = re.sub(r"([A-Za-z])(\d)", r"\1 \2", clean)
    clean = re.sub(r"(\d)([A-Za-z])", r"\1 \2", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    clean = re.sub(r"^(?:module|unit|m)\s*\d+\s*[-:]\s*", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"^\d+\s*[-:]\s*", "", clean)
    clean = re.sub(r"\b(?:module|unit)\s*[-:]?\s*\d+\b", " ", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\b(?:ktu|university|semester|scheme|regulation|course|credit|credits)\b", " ", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\b[A-Z]{2,10}\s*[-]?\s*\d{2,3}\b", " ", clean)
    clean = re.sub(r"^definition\s+of\s+", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"^\(?[a-zA-Z0-9ivxIVX]{1,4}[\).:-]\s*", "", clean)
    clean = re.sub(r"^[a-zA-Z]\s+", "", clean)
    clean = re.sub(r"^define\s+list\s+of\s+", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"^basic\s+concepts\s+of\s+", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"^general\s+idea\s+about\s+", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"^the\s+real\s+excitement\s+surrounding\s+", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"^real\s+excitement\s+surrounding\s+", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"^list\s+of\s+", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"^define\s+", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"^briefly\s+explain\s+", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"^explain\s+", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"^discuss\s+", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"^describe\s+", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"^state\s+", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"^concepts\s+of\s+", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"^fundamental\s+concepts\s+of\s+", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"^concepts\s+", "", clean, flags=re.IGNORECASE)
    clean = re.sub(
        r"^illustrate\s+practical\s+applications\s+of\s+",
        "",
        clean,
        flags=re.IGNORECASE
    )
    clean = re.sub(r"^(analyze|analyse)\s+", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"^critically\s+evaluate\s+", "", clean, flags=re.IGNORECASE)
    clean = re.sub(
        r"^propose\s+a\s+suitable\s+strategy\s+for\s+",
        "",
        clean,
        flags=re.IGNORECASE
    )
    clean = re.sub(
        r"^design\s+and\s+justify\s+an\s+approach\s+for\s+",
        "",
        clean,
        flags=re.IGNORECASE
    )
    clean = re.sub(
        r"^apply\s+the\s+principles\s+of\s+",
        "",
        clean,
        flags=re.IGNORECASE
    )
    clean = re.sub(r"\bvarious\s+", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"^(its|their|these|those|this|that)\s+", "", clean, flags=re.IGNORECASE)
    clean = re.sub(
        r"\s+(in\s+brief|in\s+detail|with\s+suitable\s+examples|to\s+an\s+appropriate\s+case)\.?$",
        "",
        clean,
        flags=re.IGNORECASE
    )
    clean = re.sub(
        r"\b(in\s+brief|in\s+detail|with\s+suitable\s+examples|to\s+an\s+appropriate\s+case)\b",
        "",
        clean,
        flags=re.IGNORECASE
    )
    clean = re.sub(r"\bwith regard to\b", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\blike\s*$", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bcertain action\b", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\beffectiveaddress\b", "effective address", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bin(this|that|the|case)\b", r"in \1", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bbasedindexed\b", "based indexed", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\banimmediate\b", "an immediate", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bi\/?o\s+\w+\s+n\s+i\/?o\s+\w+\s*\d*\b", "I/O devices", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bi\/?o\s+device\s+\d+\b", "I/O devices", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bnon\s*-\s*conventional\b", "non-conventional", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bhigh\s*-\s*power\b", "high-power", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\blow\s*-\s*power\b", "low-power", clean, flags=re.IGNORECASE)
    clean = re.sub(
        r"\bfirst\s+(?:in\s+)?first\s+out\s+fashion\b",
        "FIFO principle",
        clean,
        flags=re.IGNORECASE
    )
    clean = re.sub(
        r"\bfirst\s+out\s+fashion\b",
        "FIFO principle",
        clean,
        flags=re.IGNORECASE
    )
    clean = re.sub(
        r"\bfunctional\s+units?\s+basic\s+operational\b",
        "functional units and basic operations",
        clean,
        flags=re.IGNORECASE
    )
    clean = re.sub(
        r"\binter\s+register\s+transfer\s+arithmetic\b",
        "inter-register transfer and arithmetic operations",
        clean,
        flags=re.IGNORECASE
    )
    clean = re.sub(r"\bnumber\s+unsigned\s+multiplication\b", "unsigned multiplication", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bbinary\s+numbers?\s+array\b", "binary numbers", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bbus\s+structures?\s+memory\b", "memory bus structures", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bcpu\s+organization\s+horizontal\b", "horizontal microprogramming", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\btransfer\s+arithmetic\s+logic\b", "arithmetic logic unit and register transfer", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bsequencing\s+addressing\s+modes\b", "addressing modes and sequencing", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bdimensional\s+logic\s+array\b", "programmable logic array", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\s*-\s*", " - ", clean)
    clean = re.sub(r"\b\d+\b$", "", clean)
    clean = re.sub(r"\s+", " ", clean).strip(" .,:;\"'")
    return clean


def _title_case_if_needed(topic):

    if not topic:
        return topic
    words = topic.split()
    if words and all(word.isupper() for word in words if any(ch.isalpha() for ch in word)):
        return topic.title()
    return topic


def _is_fragment_like(text):

    if not text:
        return True

    starts = {
        "enough",
        "cause",
        "causes",
        "causing",
        "absorb",
        "absorbs",
        "absorbing",
        "prevent",
        "prevents",
        "increases",
        "decreases",
    }
    first = text.split()[0].lower()
    if first in starts:
        return True

    lowered = text.lower()
    fragment_markers = [
        " to cause ",
        " causing ",
        " from the blood ",
        " weather changes",
        " as input or as output ports",
        " upward compatible",
    ]
    return any(marker in lowered for marker in fragment_markers)


def _has_invalid_topic_structure(topic):

    text = re.sub(r"\s+", " ", str(topic or "")).strip()
    if not text:
        return True

    lowered = text.lower()
    words = lowered.split()
    if not words:
        return True

    invalid_starts = {
        "as", "and", "or", "of", "to", "in", "on", "with", "by",
        "briefly", "explain", "discuss", "describe", "state",
        "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
    }
    invalid_endings = {
        "and", "or", "of", "to", "in", "on", "with", "by", "compatible", "as",
        "what", "when", "where", "which", "who", "whom", "whose", "why", "how", "itself",
    }
    if words[0] in invalid_starts or words[-1] in invalid_endings:
        return True

    if lowered in {"briefly", "explain", "briefly explain", "discuss", "describe", "state"}:
        return True

    bad_patterns = [
        r"\bas\s+input\s+or\s+as\s+output\s+ports?\b",
        r"\b(?:\d+(?:\.\d+)?)\s*mhz\s+clock\b",
        r"\bupward\s+compatible\b",
        r"\bexecution\s+of\s+input\s+or\s+output\s+instructions\b",
    ]
    return any(re.search(pattern, lowered) for pattern in bad_patterns)


def _topic_is_list_like(topic):

    separators = [",", "/", "&"]
    if any(separator in topic for separator in separators):
        return True
    return " and " in topic.lower() and len(topic.split()) <= 6


def _is_weak_topic(topic):

    if not topic:
        return True

    lowered = topic.lower().strip()
    canonical = re.sub(r"\s*-\s*", "-", lowered)
    canonical = re.sub(r"\s+", " ", canonical).strip()
    question_words = {"what", "when", "where", "which", "who", "whom", "whose", "why", "how"}
    words = canonical.split()
    if any(word in question_words for word in words):
        return True
    if canonical.endswith("itself"):
        return True
    if len(words) >= 3 and any(word in {"find", "finds", "found", "finding"} for word in words):
        return True
    weak_exact = {
        "its effects",
        "its effect",
        "its sources",
        "its source",
        "its utilisation",
        "its utilization",
        "non-conventional",
        "conventional",
        "environmental studies",
        "evolution of the concept",
        "action",
        "certain action",
        "community",
        "key features",
        "object oriented programming",
        "object oriented programming using java",
        "credit course",
        "course outcomes",
        "high power",
        "high-power",
        "low power",
        "low-power",
        "clean development",
        "sustainable",
        "development goals",
        "committee",
        "committee on",
        "committee for",
        "committee of",
        "committee to",
        "committee in",
        "commission",
        "commission on",
        "board on",
        "council on",
        "programming lab",
        "laboratory",
        "lab",
        "real excitement surrounding",
        "first out fashion",
        "fashion",
        "while the",
        "they are trap",
        "in the above example",
        "inthe above example",
    }
    if lowered in weak_exact or canonical in weak_exact:
        return True

    weak_starts = (
        "its ",
        "their ",
        "these ",
        "those ",
        "this ",
        "that ",
        "as ",
        "of ",
        "in ",
        "on ",
        "to ",
        "for ",
        "with ",
        "by ",
    )
    if lowered.startswith(weak_starts):
        return True

    weak_contains = (
        "credit course",
        "course outcome",
        "course outcomes",
        "key features",
        "key differences between",
        "syllabus",
        "question paper",
        "is applied",
        "after each",
        "computed as",
        "disable using",
        "decodes instruction",
        "as responsible",
        "above you wants from you",
        "basic structure of computers",
        "architecture module",
        "i/o device n i/o device",
        "real excitement surrounding",
        "programming lab",
        "lab manual",
        "laboratory session",
        "first out fashion",
        "lines each",
        "three groups of",
        "while the",
        "they are trap",
        "in the above example",
        "inthe above example",
        "programmed in three different modes",
        "interfaced together to system bus",
        "address on to the stack",
    )
    if any(item in lowered for item in weak_contains):
        return True

    if re.search(r"\b[A-Z]{2,10}\s*\d{2,3}\b", topic):
        return True
    if re.search(r"\b\d+[A-Za-z]{2,}\b", topic):
        return True

    if re.match(r"^(committee|commission|board|council)\s+(on|for|of|to|in)$", canonical):
        return True

    if re.search(r"\b(on|of|for|to|in|with|by|and|or)$", canonical):
        return True
    if re.search(r"\brequires\s+\d+\s*v\b", canonical):
        return True
    if "&" in canonical:
        return True
    if re.search(r"[a-z]{12,}\d+[a-z0-9]*", canonical):
        return True
    if re.search(r"[A-Z]{4,}[A-Z][a-z]{3,}", topic):
        return True
    if re.search(r"[a-z]{20,}", canonical):
        return True
    if re.search(r"[a-z]+[A-Z]{2,}", topic):
        return True
    if re.search(r"[A-Z]{2,}[a-z]{3,}", topic):
        return True
    if re.search(r"[A-Z]{2,}[A-Z][a-z]{2,}", topic):
        return True
    if re.search(r"\b(?:which|that|the|or|and|to|in|on|of|for|with|by)(?:contains|operand|location|register|stands|serial|data|address)\b", canonical):
        return True
    if re.search(r"\b(while|they|it|this|that)\b", canonical):
        return True
    if re.search(r"\b(programmed|interfaced)\s+in\b", canonical):
        return True
    if re.search(
        r"\b(contains|contain|stands for|may be|provide|require|referred using|using the|"
        r"program counter to zero|address of the memory|execution of input or output instructions)\b",
        canonical
    ):
        return True
    if re.search(r"\bcalled as\b", canonical):
        return True
    if _has_invalid_topic_structure(topic):
        return True
    if re.search(r"\b(?:\d+(?:\.\d+)?)\s*mhz\b", canonical):
        return True
    if re.search(r"\b\d+\s*h\s+is\b", canonical):
        return True
    if re.search(r"\bfirst completes\b", canonical):
        return True
    if any(len(token) >= 16 for token in canonical.split()):
        return True
    tokens_title = [t for t in topic.split() if t]
    if len(tokens_title) == 3 and tokens_title[0].istitle() and tokens_title[1].istitle():
        if len(tokens_title[1]) <= 3 and len(tokens_title[2]) <= 4:
            return True

    tokens = [token for token in re.split(r"[^a-z]+", canonical) if token]
    if len(tokens) > 7:
        return True
    clause_markers = {"is", "are", "was", "were", "be", "being", "contains", "contain", "provide", "require", "referred", "using", "may"}
    if len(tokens) > 4 and any(token in clause_markers for token in tokens):
        return True
    if len(tokens) == 2 and tokens[1] in {"lab", "laboratory"}:
        return True
    if len(tokens) == 2:
        qualifier_tokens = {"high", "low", "basic", "advanced", "general", "clean", "non"}
        generic_tokens = {
            "power", "conventional", "technology", "technologies", "method", "methods",
            "system", "systems", "development", "applications", "standards"
        }
        if tokens[0] in qualifier_tokens and tokens[1] in generic_tokens:
            return True

    if len(topic.split()) == 1:
        allowed_single = {
            "sustainability",
            "pollution",
            "biodiversity",
            "ecosystem",
            "ozone",
            "lead",
            "inheritance",
            "polymorphism",
            "encapsulation",
            "abstraction",
            "package",
            "packages",
            "thread",
            "threads",
            "multithreading",
            "synchronization",
            "interface",
            "interfaces",
            "class",
            "classes",
            "object",
            "objects",
            "constructor",
            "constructors",
            "overloading",
            "overriding",
            "exception",
            "exceptions",
            "swing",
            "awt",
            "gui",
        }
        if lowered not in allowed_single:
            return True

    return False


def _normalize_ethics_topic(topic):

    lowered = topic.lower()
    if "kohlberg" in lowered and "gilligan" in lowered:
        return "Kohlberg's and Gilligan's moral development theories"
    if "moral autonomy" in lowered:
        return "moral autonomy in engineering ethics"
    if "moral dilemmas" in lowered:
        return "moral dilemmas in engineering practice"
    if "gas tragedy" in lowered or "bhopal" in lowered:
        return "Bhopal gas tragedy"
    if "professional engineering bodies" in lowered:
        return "professional engineering bodies"
    if "multinational corporations" in lowered and "environmental ethics" in lowered:
        return "environmental ethics in multinational corporations"
    return topic


def _quality_checked_topic(topic):

    clean = _normalize_ethics_topic(_finalize_topic(topic))
    if _is_weak_topic(clean):
        return ""
    return clean


def _preferred_blooms_for_section(section_marks, selected_levels):

    if section_marks <= 3:
        preferred = ["Remember", "Understand", "Apply"]
    else:
        preferred = ["Apply", "Analyze", "Evaluate", "Create"]

    ordered = [level for level in preferred if level in selected_levels]
    if ordered:
        return ordered
    return selected_levels


def _topic_tokens(topic):

    return [token for token in re.split(r"[^A-Za-z]+", topic.lower()) if token]


def _classify_topic(topic):

    lowered = topic.lower()
    tokens = set(_topic_tokens(topic))

    compare_markers = {"cyclones", "hurricanes", "solar", "wind", "water", "air"}
    process_markers = {
        "management", "assessment", "corrosion", "drainage",
        "mitigation", "adaptation", "treatment", "utilisation", "utilization"
    }
    cause_effect_markers = {
        "pollution", "disaster", "disasters", "burst", "wave", "warming",
        "hazard", "hazards", "flood", "floods", "climate", "effect", "effects"
    }
    application_markers = {
        "technology", "cells", "energy", "plants", "mechanism", "standards",
        "ecology", "symbiosis", "goals", "development"
    }

    if _topic_is_list_like(topic):
        if " and " in lowered or "," in topic:
            return "comparison"
        return "listing"
    if tokens & process_markers:
        return "process"
    if tokens & cause_effect_markers:
        return "cause_effect"
    if tokens & application_markers:
        return "application"
    if tokens & compare_markers and len(tokens) > 1:
        return "comparison"
    return "concept"


def _topic_without_plural_disasters(topic):

    clean = re.sub(r"\bdisasters\b", "disaster", topic, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", clean).strip()


def _topic_has_any(topic, keywords):

    lowered = topic.lower()
    return any(keyword in lowered for keyword in keywords)


def _render_domain_specific_short(topic, bloom):

    if _topic_has_any(topic, ["professional ethics", "engineering ethics"]):
        prompts = {
            "Remember": f"Define {topic}.",
            "Understand": f"Explain the importance of {topic}.",
            "Apply": f"State applications of {topic} in practice.",
        }
        return prompts.get(bloom)

    if _topic_has_any(topic, ["moral autonomy"]):
        prompts = {
            "Remember": "Define moral autonomy.",
            "Understand": "Explain moral autonomy in engineering decisions.",
            "Apply": "State examples of moral autonomy in engineering practice.",
        }
        return prompts.get(bloom)

    if _topic_has_any(topic, ["kohlberg", "gilligan"]):
        prompts = {
            "Remember": "Define Kohlberg's moral development theory.",
            "Understand": "Differentiate Kohlberg's and Gilligan's moral development theories.",
            "Apply": "State relevance of Kohlberg's and Gilligan's theories to ethics.",
        }
        return prompts.get(bloom)

    if _topic_has_any(topic, ["bhopal gas tragedy", "gas tragedy"]):
        prompts = {
            "Remember": "Define Bhopal gas tragedy.",
            "Understand": "Explain causes of the Bhopal gas tragedy.",
            "Apply": "State key lessons from the Bhopal gas tragedy.",
        }
        return prompts.get(bloom)

    if _topic_has_any(topic, ["professional engineering bodies"]):
        prompts = {
            "Remember": "Define professional engineering bodies.",
            "Understand": "Explain the role of professional engineering bodies.",
            "Apply": "State functions of professional engineering bodies.",
        }
        return prompts.get(bloom)

    if _topic_has_any(topic, ["standards", "ems"]):
        prompts = {
            "Remember": f"Define {topic}.",
            "Understand": f"Explain the importance of {topic}.",
            "Apply": f"State the applications of {topic}.",
        }
        return prompts.get(bloom)

    if _topic_has_any(topic, ["goals"]):
        prompts = {
            "Remember": f"Define {topic}.",
            "Understand": f"Explain the significance of {topic}.",
            "Apply": f"State the key objectives of {topic}.",
        }
        return prompts.get(bloom)

    if _topic_has_any(topic, ["plants", "fuel cells", "wind energy", "solar energy", "geothermal", "hydro"]):
        prompts = {
            "Remember": f"Define {topic}.",
            "Understand": f"Explain the working principle of {topic}.",
            "Apply": f"State the applications of {topic}.",
        }
        return prompts.get(bloom)

    if _topic_has_any(topic, ["management", "assessment"]):
        prompts = {
            "Remember": f"Define {topic}.",
            "Understand": f"Explain the major steps in {topic}.",
            "Apply": f"Illustrate the application of {topic}.",
        }
        return prompts.get(bloom)

    if _topic_has_any(topic, ["cloud burst", "heat wave", "cold wave", "sea erosion", "avalanches"]):
        prompts = {
            "Remember": f"Define {topic}.",
            "Understand": f"Explain the causes of {topic}.",
            "Apply": f"State the effects of {topic}.",
        }
        return prompts.get(bloom)

    return None


def _render_domain_specific_long(topic, bloom):

    if _topic_has_any(topic, ["moral dilemmas"]):
        prompts = {
            "Apply": "Discuss moral dilemmas in engineering practice with suitable examples.",
            "Analyze": "Analyze moral dilemmas in engineering practice with suitable examples.",
            "Evaluate": "Critically evaluate approaches to resolve moral dilemmas with suitable examples.",
            "Create": "Propose a framework to resolve moral dilemmas in engineering practice with suitable examples.",
        }
        return prompts.get(bloom)

    if _topic_has_any(topic, ["multinational corporations", "environmental ethics"]):
        prompts = {
            "Apply": "Explain environmental ethics in multinational corporations with suitable examples.",
            "Analyze": "Analyze environmental ethics issues in multinational corporations with suitable examples.",
            "Evaluate": "Critically evaluate environmental responsibilities of multinational corporations with suitable examples.",
            "Create": "Design an ethical policy for multinational corporations on environmental issues with suitable examples.",
        }
        return prompts.get(bloom)

    if _topic_has_any(topic, ["professionalism", "professional ethics"]):
        prompts = {
            "Apply": "Discuss professionalism in engineering practice with suitable examples.",
            "Analyze": "Analyze models of professionalism in engineering practice with suitable examples.",
            "Evaluate": "Critically evaluate professionalism models in engineering practice with suitable examples.",
            "Create": "Develop a professional code of conduct for engineers with suitable examples.",
        }
        return prompts.get(bloom)

    if _topic_has_any(topic, ["standards", "ems"]):
        prompts = {
            "Apply": f"Explain the applications of {topic} with suitable examples.",
            "Analyze": f"Analyze the role of {topic} in environmental management with suitable examples.",
            "Evaluate": f"Critically evaluate {topic} with suitable examples.",
            "Create": f"Design a suitable implementation plan for {topic} with suitable examples.",
        }
        return prompts.get(bloom)

    if _topic_has_any(topic, ["goals"]):
        prompts = {
            "Apply": f"Explain the major goals of {topic} with suitable examples.",
            "Analyze": f"Analyze the importance of {topic} with suitable examples.",
            "Evaluate": f"Critically evaluate the implementation of {topic} with suitable examples.",
            "Create": f"Suggest a suitable action plan to achieve {topic} with suitable examples.",
        }
        return prompts.get(bloom)

    if _topic_has_any(topic, ["plants", "fuel cells", "wind energy", "solar energy", "geothermal", "hydro"]):
        prompts = {
            "Apply": f"Explain the applications of {topic} with suitable examples.",
            "Analyze": f"Analyze the working and applications of {topic} with suitable examples.",
            "Evaluate": f"Critically evaluate {topic} with suitable examples.",
            "Create": f"Design and justify a suitable energy system using {topic} with suitable examples.",
        }
        return prompts.get(bloom)

    if _topic_has_any(topic, ["management", "assessment"]):
        prompts = {
            "Apply": f"Explain the procedure involved in {topic} with suitable examples.",
            "Analyze": f"Analyze the major steps in {topic} with suitable examples.",
            "Evaluate": f"Critically evaluate the effectiveness of {topic} with suitable examples.",
            "Create": f"Design and justify an improved approach for {topic} with suitable examples.",
        }
        return prompts.get(bloom)

    if _topic_has_any(topic, ["cloud burst", "heat wave", "cold wave", "sea erosion", "avalanche", "disaster"]):
        prompts = {
            "Apply": f"Discuss the impacts of {topic} with suitable examples.",
            "Analyze": f"Analyze the causes and effects of {topic} with suitable examples.",
            "Evaluate": f"Critically evaluate the management of {topic} with suitable examples.",
            "Create": f"Suggest suitable mitigation measures for {topic} with suitable examples.",
        }
        return prompts.get(bloom)

    return None


def _choose_list_stem(bloom, is_long):

    short_stems = {
        "Remember": "List",
        "Understand": "Explain",
        "Apply": "Illustrate applications of",
        "Analyze": "Analyze",
        "Evaluate": "Evaluate",
        "Create": "Suggest measures for",
    }
    long_stems = {
        "Remember": "Define and explain",
        "Understand": "Explain",
        "Apply": "Apply the principles of",
        "Analyze": "Analyze",
        "Evaluate": "Critically evaluate",
        "Create": "Design and justify an approach for",
    }
    stems = long_stems if is_long else short_stems
    return stems.get(bloom, "Explain")


def _finalize_topic(topic):

    clean = _title_case_if_needed(_sanitize_topic(topic))
    clean = re.sub(r"^(?:the|a|an)\s+", "", clean, flags=re.IGNORECASE)
    return clean or "Environmental Studies"


def _module_concept_map(module_concepts):

    concept_map = {}
    for module in module_concepts or []:
        module_id = int(module.get("module", 1))
        items = []
        seen = set()
        for raw in module.get("concepts", []):
            clean = _quality_checked_topic(raw)
            if not clean:
                continue
            if re.search(r"\b[A-Z]{2,5}\s*\d{2,3}\b", clean):
                continue
            key = clean.lower()
            if key in seen:
                continue
            seen.add(key)
            items.append(clean)
        if items:
            concept_map[module_id] = items
    return concept_map


def _pick_concept_for_module(module, concept_map, fallback_concepts, usage_counts):

    module_items = concept_map.get(module, [])
    if module_items:
        index = usage_counts.get(module, 0) % len(module_items)
        usage_counts[module] = usage_counts.get(module, 0) + 1
        return module_items[index]

    if fallback_concepts:
        index = sum(usage_counts.values()) % len(fallback_concepts)
        return fallback_concepts[index]

    return "Environmental Studies"


def _pick_quality_concept(module, concept_map, fallback_concepts, usage_counts):

    start_module_count = usage_counts.get(module, 0)
    attempts = len(concept_map.get(module, []))
    for _ in range(attempts):
        concept = _pick_concept_for_module(module, concept_map, fallback_concepts, usage_counts)
        if not _is_weak_topic(concept):
            return concept
    usage_counts[module] = start_module_count

    for concept in fallback_concepts:
        if not _is_weak_topic(concept):
            return concept

    return "Environmental pollution"


def _is_short_answer_style(question_text):

    allowed = (
        "Define ",
        "List ",
        "State ",
        "Name ",
        "Explain ",
        "Briefly explain ",
        "Differentiate ",
        "Outline ",
        "Illustrate ",
    )
    blocked = (
        "Critically evaluate ",
        "Design and justify ",
        "Discuss ",
        "Evaluate ",
        "Propose a suitable strategy ",
    )
    return question_text.startswith(allowed) and not question_text.startswith(blocked)


def _is_long_answer_style(question_text):

    allowed = (
        "Explain ",
        "Discuss ",
        "Analyze ",
        "Critically evaluate ",
        "Design and justify ",
        "Apply the principles of ",
        "Define and explain ",
        "Compare ",
    )
    return question_text.startswith(allowed)


def _render_short_question(topic, bloom, topic_class):

    subject = _topic_without_plural_disasters(topic)
    specific = _render_domain_specific_short(topic, bloom)
    if specific:
        return specific

    if topic_class == "comparison":
        prompts = {
            "Remember": f"List the types of {topic}.",
            "Understand": f"Differentiate between {topic}.",
            "Apply": f"Illustrate the practical significance of {topic}.",
        }
        return prompts.get(bloom, f"Explain {topic} in brief.")

    if topic_class == "process":
        prompts = {
            "Remember": f"Define {topic}.",
            "Understand": f"Outline the steps in {topic}.",
            "Apply": f"Illustrate the application of {topic}.",
        }
        return prompts.get(bloom, f"Explain {topic} in brief.")

    if topic_class == "cause_effect":
        prompts = {
            "Remember": f"Define {subject}.",
            "Understand": f"Explain the major causes of {subject}.",
            "Apply": f"State the effects of {subject}.",
        }
        return prompts.get(bloom, f"Explain {subject} in brief.")

    if topic_class == "application":
        prompts = {
            "Remember": f"Define {topic}.",
            "Understand": f"Explain the principle of {topic}.",
            "Apply": f"State the applications of {topic}.",
        }
        return prompts.get(bloom, f"Explain {topic} in brief.")

    prompts = {
        "Remember": f"Define {topic}.",
        "Understand": f"Explain {topic} in brief.",
        "Apply": f"State the applications of {topic}.",
    }
    return prompts.get(bloom, f"Explain {topic} in brief.")


def _render_long_question(topic, bloom, topic_class):

    subject = _topic_without_plural_disasters(topic)
    specific = _render_domain_specific_long(topic, bloom)
    if specific:
        return specific

    if topic_class == "comparison":
        prompts = {
            "Apply": f"Compare {topic} with suitable examples.",
            "Analyze": f"Analyze the key differences between {topic} with suitable examples.",
            "Evaluate": f"Critically evaluate {topic} with suitable examples.",
            "Create": f"Design and justify an approach for managing issues related to {topic} with suitable examples.",
        }
        return prompts.get(bloom, f"Discuss {topic} with suitable examples.")

    if topic_class == "process":
        prompts = {
            "Apply": f"Explain the procedure involved in {topic} with suitable examples.",
            "Analyze": f"Analyze the process of {topic} with suitable examples.",
            "Evaluate": f"Critically evaluate the effectiveness of {topic} with suitable examples.",
            "Create": f"Design and justify an approach for improving {topic} with suitable examples.",
        }
        return prompts.get(bloom, f"Discuss {topic} with suitable examples.")

    if topic_class == "cause_effect":
        prompts = {
            "Apply": f"Discuss the impacts of {subject} with suitable examples.",
            "Analyze": f"Analyze the causes and effects of {subject} with suitable examples.",
            "Evaluate": f"Critically evaluate the management of {subject} with suitable examples.",
            "Create": f"Suggest suitable mitigation strategies for {subject} with suitable examples.",
        }
        return prompts.get(bloom, f"Discuss {subject} with suitable examples.")

    if topic_class == "application":
        prompts = {
            "Apply": f"Explain the applications of {topic} with suitable examples.",
            "Analyze": f"Analyze the role of {topic} in practice with suitable examples.",
            "Evaluate": f"Critically evaluate {topic} with suitable examples.",
            "Create": f"Design and justify a suitable plan based on {topic} with suitable examples.",
        }
        return prompts.get(bloom, f"Discuss {topic} with suitable examples.")

    prompts = {
        "Apply": f"Explain the applications of {topic} with suitable examples.",
        "Analyze": f"Analyze {topic} with suitable examples.",
        "Evaluate": f"Critically evaluate {topic} with suitable examples.",
        "Create": f"Discuss {topic} with suitable examples.",
    }
    return prompts.get(bloom, f"Discuss {topic} with suitable examples.")


def _comparison_parts(subject):

    parts = re.split(r"\s+(?:and|vs|versus)\s+|/|,", subject, flags=re.IGNORECASE)
    cleaned = [part.strip(" .,:;") for part in parts if part.strip(" .,:;")]
    return cleaned


def _normalize_articles(text):

    updated = re.sub(r"\ba ([aeiouAEIOU][A-Za-z-]*)\b", r"an \1", text)
    updated = re.sub(r"\ban ([^aeiouAEIOU\W][A-Za-z-]*)\b", r"a \1", updated)
    return updated


def _naturalize_question_text(question_text):

    text = re.sub(r"\s+", " ", str(question_text or "")).strip()
    if not text:
        return ""

    replacements = [
        (r"^Explain (.+?) in brief\.$", r"Briefly explain \1."),
        (
            r"\ba suitable implementation plan for\b",
            "an implementation plan for",
        ),
        (r"\ba suitable action plan to achieve\b", "an action plan to achieve"),
        (r"\ba suitable plan based on\b", "an effective plan based on"),
        (r"\bthe procedure involved in\b", "the procedure for"),
        (r"\binter\s*-\s*register\b", "inter-register"),
    ]
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

    text = _naturalize_comparison_question_text(text)
    text = _normalize_articles(text)
    text = re.sub(r"\b(\w+)(\s+\1\b)+", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()

    if text and text[-1] not in ".?!":
        text += "."

    if text:
        text = text[0].upper() + text[1:]

    return text


def _inject_question_variation(question_text, is_short):

    text = re.sub(r"\s+", " ", str(question_text or "")).strip()
    if not text:
        return ""

    short_variants = [
        (r"^State the applications of (.+?)\.$", "List the applications of {topic}."),
        (r"^Define (.+?)\.$", "Briefly explain {topic}."),
        (r"^Briefly explain (.+?)\.$", "Explain {topic} in brief."),
    ]
    long_variants = [
        (
            r"^Explain the applications of (.+?) with suitable examples\.$",
            "Discuss the applications of {topic} with suitable examples."
        ),
        (
            r"^Analyze (.+?) with suitable examples\.$",
            "Discuss {topic} with suitable examples."
        ),
    ]

    candidates = short_variants if is_short else long_variants
    for pattern, template in candidates:
        match = re.match(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        if random.random() < 0.35:
            topic = match.group(1).strip(" .,:;")
            variant = template.format(topic=topic)
            return _naturalize_question_text(variant)
        break

    return text


def _looks_like_comparison_subject(subject):

    lowered = subject.lower()
    markers = [" and ", " vs ", " versus ", "/", ","]
    return any(marker in lowered for marker in markers)


def _naturalize_comparison_question_text(text):

    between_match = re.match(r"^Differentiate between (.+?)\.$", text, flags=re.IGNORECASE)
    if between_match:
        subject = between_match.group(1).strip()
        parts = _comparison_parts(subject)
        if len(parts) >= 2:
            return f"Differentiate between {parts[0]} and {parts[1]}."
        if not _looks_like_comparison_subject(subject):
            return f"Explain key aspects of {subject}."
        return text

    compare_match = re.match(
        r"^Compare (.+?) with suitable examples\.$",
        text,
        flags=re.IGNORECASE
    )
    if compare_match:
        subject = compare_match.group(1).strip()
        parts = _comparison_parts(subject)
        if len(parts) >= 2:
            return f"Compare {parts[0]} and {parts[1]} with suitable examples."
        if not _looks_like_comparison_subject(subject):
            return f"Discuss {subject} with suitable examples."
        return text

    analyze_compare_match = re.match(
        r"^Analyze the key differences between (.+?) with suitable examples\.$",
        text,
        flags=re.IGNORECASE
    )
    if analyze_compare_match:
        subject = analyze_compare_match.group(1).strip()
        parts = _comparison_parts(subject)
        if len(parts) >= 2:
            return (
                f"Analyze the key differences between {parts[0]} and {parts[1]} "
                "with suitable examples."
            )
        if not _looks_like_comparison_subject(subject):
            return f"Analyze {subject} with suitable examples."
        return text

    return text


def _truncate_topic_words(topic, max_words):

    words = topic.split()
    if len(words) <= max_words:
        return topic
    return " ".join(words[:max_words])


def _enforce_short_question_brevity(text, max_words=12):

    question = re.sub(r"\s+", " ", str(text or "")).strip()
    if not question:
        return ""

    words = question.rstrip(".?!").split()
    if len(words) <= max_words:
        return question

    patterns = [
        (r"^Briefly explain (.+?)\.$", "Explain"),
        (r"^Explain the major causes of (.+?)\.$", "State"),
        (r"^Explain the importance of (.+?)\.$", "State"),
        (r"^Explain the significance of (.+?)\.$", "State"),
        (r"^Explain the working principle of (.+?)\.$", "Explain"),
        (r"^Explain the principle of (.+?)\.$", "Explain"),
        (r"^Outline the steps in (.+?)\.$", "Outline"),
        (r"^Illustrate the practical significance of (.+?)\.$", "State"),
        (r"^Illustrate the application of (.+?)\.$", "State"),
        (r"^Differentiate between (.+?)\.$", "Differentiate between"),
        (r"^State the applications of (.+?)\.$", "State"),
        (r"^Define (.+?)\.$", "Define"),
        (r"^Explain (.+?)\.$", "Explain"),
        (r"^State (.+?)\.$", "State"),
    ]

    for pattern, stem in patterns:
        match = re.match(pattern, question, flags=re.IGNORECASE)
        if not match:
            continue
        topic = match.group(1).strip(" .,:;")
        if stem.lower() == "differentiate between":
            parts = _comparison_parts(topic)
            if len(parts) >= 2:
                concise = f"Differentiate between {parts[0]} and {parts[1]}."
                if len(concise.rstrip(".?!").split()) <= max_words:
                    return concise
            topic = _truncate_topic_words(topic, max_words - 2)
            return f"Explain {topic}."

        budget = max(1, max_words - len(stem.split()))
        compact_topic = _truncate_topic_words(topic, budget)
        concise = f"{stem} {compact_topic}."
        if len(concise.rstrip(".?!").split()) <= max_words:
            return concise

    fallback_topic = _truncate_topic_words(
        question.rstrip(".?!"),
        max_words - 1
    )
    return f"Define {fallback_topic}."


def _build_short_question_text(concept, bloom):

    topic = _quality_checked_topic(concept) or "Environmental pollution"
    topic_class = _classify_topic(topic)
    if _is_fragment_like(topic):
        topic_class = "concept"
    text = _naturalize_question_text(_render_short_question(topic, bloom, topic_class))
    return _enforce_short_question_brevity(text, max_words=12)


def _build_long_question_text(concept, bloom):

    topic = _quality_checked_topic(concept) or "Environmental pollution"
    topic_class = _classify_topic(topic)
    if _is_fragment_like(topic):
        topic_class = "concept"
    return _naturalize_question_text(_render_long_question(topic, bloom, topic_class))


def _is_low_quality_question(question_text):

    text = re.sub(r"\s+", " ", str(question_text or "")).strip()
    lowered = text.lower()
    if not text:
        return True
    tokens = [token for token in re.split(r"[^a-z]+", lowered) if token]
    question_words = {"what", "when", "where", "which", "who", "whom", "whose", "why", "how"}
    if tokens and (tokens[-1] in question_words or tokens[-1] in {"itself", "found", "finding"}):
        return True
    if len(tokens) >= 3 and len(set(tokens)) < len(tokens):
        return True

    bad_markers = [
        "certain action",
        " with regard to ",
        " like.",
        " and based on the rich experience",
        "professionalism- models of professional",
        "ktu",
        "module",
        "course",
        "credit",
        "question paper",
        "functional units basic operational",
        "inter register transfer arithmetic",
        "real excitement surrounding",
        "programming lab",
        "first out fashion",
        "while the",
        "they are trap",
        "in the above example",
        "inthe above example",
        "lines each",
        "three groups of",
        "programmed in three different modes",
        "interfaced together to system bus",
        "address on to the stack",
        "concepts instruction",
        "numbers array",
        "organization horizontal",
        "bus structures memory",
        "transfer arithmetic logic",
        "number unsigned multiplication",
        "dimensional logic array",
        "sequencing addressing modes",
    ]
    if any(marker in lowered for marker in bad_markers):
        return True

    if len(text.rstrip(".?!").split()) < 4:
        return True

    if re.search(r"\b(on|of|for|to|in|with|by|and|or)\.$", text, flags=re.IGNORECASE):
        return True
    if re.search(r"\brequires\s+\d+\s*[vV]\b", lowered):
        return True
    if re.search(r"&\s*[.?!]?$", text):
        return True
    if re.search(r"[A-Za-z]{12,}\d+[A-Za-z0-9]*", text):
        return True
    if re.search(r"[A-Z]{4,}[A-Z][a-z]{3,}", text):
        return True
    if re.search(r"[a-z]{20,}", lowered):
        return True
    if re.search(r"[a-z]+[A-Z]{2,}", text):
        return True
    if re.search(r"[A-Z]{2,}[a-z]{3,}", text):
        return True
    if re.search(r"[A-Z]{2,}[A-Z][a-z]{2,}", text):
        return True
    if re.search(r"\b(?:which|that|the|or|and|to|in|on|of|for|with|by)(?:contains|operand|location|register|stands|serial|data|address)\b", lowered):
        return True
    if re.search(
        r"\b(contains one|stands for|may be defined|provide or require|referred using|using the particular|"
        r"program counter to zero|address of the memory ?location|execution of input or output instructions)\b",
        lowered
    ):
        return True
    if re.search(r"\b(as input or as output ports|upward compatible)\b", lowered):
        return True
    if re.search(r"\b(?:\d+(?:\.\d+)?)\s*mhz\s+clock\b", lowered):
        return True
    if re.search(r"\bcalled as\b", lowered):
        return True
    if re.search(
        r"^(define|list|state|explain|briefly explain|discuss|analyze|analyse|evaluate|critically evaluate)\s+(of|in|on|to|for|with|by)\b",
        lowered
    ):
        return True

    return False


def _question_uniqueness_key(question_text):

    normalized = re.sub(r"[^a-z0-9]+", " ", str(question_text or "").lower())
    return re.sub(r"\s+", " ", normalized).strip()


def _question_topic_key(question_text):

    text = re.sub(r"\s+", " ", str(question_text or "")).strip().lower()
    if not text:
        return ""

    text = text.rstrip(".?!")
    text = re.sub(
        r"^(define(?: and explain)?|briefly explain|explain|describe|state(?: the)?(?: applications of)?|"
        r"list|name|outline|illustrate(?: the)?(?: applications of)?|differentiate between|"
        r"discuss|analyze|analyse|evaluate|critically evaluate|compare|apply the principles of)\s+",
        "",
        text
    )
    text = re.sub(r"\bwith suitable examples\b", "", text)
    text = re.sub(r"\bin brief\b", "", text)
    text = re.sub(r"\bin detail\b", "", text)
    text = re.sub(r"\bthe key differences between\b", "", text)
    text = re.sub(
        r"^(?:the\s+)?(?:applications?|importance|significance|principle|working principle|"
        r"major causes|causes|effects|impacts|role|procedure|steps)\s+of\s+",
        "",
        text
    )
    tokens = [token for token in re.split(r"[^a-z0-9]+", text) if token]
    stop_tokens = {
        "the", "a", "an", "of", "to", "for", "in", "on", "with", "and", "or",
        "by", "from", "using", "based", "suitable", "example", "examples",
        "application", "applications", "importance", "significance", "principle",
        "principles", "working", "major", "causes", "effects", "impacts",
        "role", "procedure", "steps", "key", "differences", "between",
    }
    normalized_tokens = []
    for token in tokens:
        if token in stop_tokens:
            continue
        if len(token) > 4 and token.endswith("s"):
            token = token[:-1]
        normalized_tokens.append(token)
    return " ".join(normalized_tokens).strip()


def _unique_quality_concepts(concept_map, fallback_concepts):

    ordered = []
    seen = set()
    for module in sorted(concept_map.keys()):
        for concept in concept_map.get(module, []):
            clean = _quality_checked_topic(concept)
            if not clean:
                continue
            key = clean.lower()
            if key in seen:
                continue
            seen.add(key)
            ordered.append(clean)
    for concept in fallback_concepts:
        clean = _quality_checked_topic(concept)
        if not clean:
            continue
        key = clean.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(clean)
    return ordered


def _fallback_question_from_topic(topic, bloom, is_short):

    normalized_topic = _quality_checked_topic(topic) or "engineering ethics"
    if is_short:
        question = _build_short_question_text(normalized_topic, bloom)
        if not _is_short_answer_style(question):
            question = f"Define {normalized_topic}."
        return _enforce_short_question_brevity(question, max_words=12)

    question = _build_long_question_text(normalized_topic, bloom)
    if not _is_long_answer_style(question):
        question = f"Analyze {normalized_topic} with suitable examples."
    return question


def generate_paper_fast(
    concepts,
    paper_model,
    blooms_levels,
    module_concepts,
    paper_model_key
):

    if not concepts:
        concepts = ["the given syllabus topics"]

    concept_map = _module_concept_map(module_concepts)
    concept_map = {
        module: random.sample(items, len(items))
        for module, items in concept_map.items()
    }
    fallback_concepts = []
    seen_fallback = set()
    for concept in concepts:
        clean = _quality_checked_topic(concept)
        if not clean:
            continue
        key = clean.lower()
        if key in seen_fallback:
            continue
        seen_fallback.add(key)
        fallback_concepts.append(clean)
    module_usage = {}
    for module, items in concept_map.items():
        if items:
            module_usage[module] = random.randrange(len(items))
    all_quality_concepts = _unique_quality_concepts(concept_map, fallback_concepts)

    module_count = max(1, len(module_concepts))
    section_a_modules = []
    section_b_modules = []

    if paper_model_key == "100" and module_count >= 5:
        section_a_modules = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
        section_b_modules = [1, 2, 3, 4, 5]
        random.shuffle(section_a_modules)
        random.shuffle(section_b_modules)

    lines = []
    used_question_keys = set()
    used_topic_keys = set()
    for section in paper_model["sections"]:
        lines.append(section["name"])
        section_blooms = _preferred_blooms_for_section(
            section["marks"], blooms_levels
        )
        bloom_start = random.randrange(len(section_blooms)) if section_blooms else 0
        for idx in range(1, section["count"] + 1):
            if section["marks"] <= 3:
                if section_a_modules:
                    module = section_a_modules[idx - 1]
                else:
                    module = ((idx - 1) % module_count) + 1
            else:
                if section_b_modules:
                    module = section_b_modules[idx - 1]
                else:
                    module = ((idx - 1) % module_count) + 1

            module_pool_size = len(concept_map.get(module, []))
            max_attempts = max(3, module_pool_size + 2)
            selected_text = ""
            selected_bloom = section_blooms[(idx - 1) % len(section_blooms)]
            selected_key = ""
            accepted_unique = False

            for attempt in range(max_attempts):
                bloom = section_blooms[(bloom_start + (idx - 1) + attempt) % len(section_blooms)]
                concept = _pick_quality_concept(
                    module, concept_map, fallback_concepts, module_usage
                )

                if section["marks"] <= 3:
                    question_text = _build_short_question_text(concept, bloom)
                    if not _is_short_answer_style(question_text):
                        question_text = _fallback_question_from_topic(concept, bloom, True)
                    if _is_low_quality_question(question_text):
                        question_text = _fallback_question_from_topic(concept, bloom, True)
                    question_text = _inject_question_variation(question_text, is_short=True)
                    question_text = _enforce_short_question_brevity(question_text, max_words=12)
                    if not _is_short_answer_style(question_text):
                        question_text = _fallback_question_from_topic(concept, bloom, True)
                else:
                    question_text = _build_long_question_text(concept, bloom)
                    if not _is_long_answer_style(question_text):
                        question_text = _fallback_question_from_topic(concept, bloom, False)
                    if _is_low_quality_question(question_text):
                        question_text = _fallback_question_from_topic(concept, bloom, False)
                    question_text = _inject_question_variation(question_text, is_short=False)
                    if not _is_long_answer_style(question_text):
                        question_text = _fallback_question_from_topic(concept, bloom, False)

                key = _question_uniqueness_key(question_text)
                topic_key = _question_topic_key(question_text)
                selected_text = question_text
                selected_bloom = bloom
                selected_key = key
                if topic_key and topic_key in used_topic_keys:
                    continue
                if key and key not in used_question_keys:
                    used_question_keys.add(key)
                    if topic_key:
                        used_topic_keys.add(topic_key)
                    accepted_unique = True
                    break

            if not accepted_unique:
                for concept in all_quality_concepts:
                    bloom = section_blooms[(idx - 1) % len(section_blooms)]
                    if section["marks"] <= 3:
                        question_text = _build_short_question_text(concept, bloom)
                        if not _is_short_answer_style(question_text):
                            question_text = _fallback_question_from_topic(concept, bloom, True)
                        if _is_low_quality_question(question_text):
                            question_text = _fallback_question_from_topic(concept, bloom, True)
                        question_text = _inject_question_variation(question_text, is_short=True)
                        question_text = _enforce_short_question_brevity(question_text, max_words=12)
                        if not _is_short_answer_style(question_text):
                            question_text = _fallback_question_from_topic(concept, bloom, True)
                    else:
                        question_text = _build_long_question_text(concept, bloom)
                        if not _is_long_answer_style(question_text):
                            question_text = _fallback_question_from_topic(concept, bloom, False)
                        if _is_low_quality_question(question_text):
                            question_text = _fallback_question_from_topic(concept, bloom, False)
                        question_text = _inject_question_variation(question_text, is_short=False)
                        if not _is_long_answer_style(question_text):
                            question_text = _fallback_question_from_topic(concept, bloom, False)

                    key = _question_uniqueness_key(question_text)
                    topic_key = _question_topic_key(question_text)
                    if topic_key and topic_key in used_topic_keys:
                        continue
                    if key and key not in used_question_keys:
                        selected_text = question_text
                        selected_bloom = bloom
                        selected_key = key
                        used_question_keys.add(key)
                        if topic_key:
                            used_topic_keys.add(topic_key)
                        accepted_unique = True
                        break

            lines.append(
                f"{idx}. [Module: {module}] [Bloom: {selected_bloom}] {selected_text}"
            )
        lines.append("")

    return "\n".join(lines).strip()
