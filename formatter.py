import re


def _ensure_tags(line):

    updated = line
    if "[Module:" not in updated:
        updated = f"[Module: 1] {updated}"
    if "[Bloom:" not in updated:
        updated = re.sub(
            r"(\[Module:\s*[1-5]\])",
            r"\1 [Bloom: Understand]",
            updated,
            count=1
        )
    return updated


def _to_display_text(line):

    text = re.sub(r"\[Module:\s*[1-5]\]\s*", "", line, count=1)
    text = re.sub(
        r"\[Bloom:\s*(Remember|Understand|Apply|Analyze|Evaluate|Create)\]\s*",
        "",
        text,
        count=1
    )
    return text.strip()


def _repair_question_text(text):

    repaired = re.sub(r"\s+", " ", str(text or "")).strip()
    if not repaired:
        return ""

    replacements = [
        (
            r"\bfunctional units basic operational\b",
            "functional units and basic operations",
        ),
        (
            r"\binter register transfer arithmetic\b",
            "inter-register transfer and arithmetic operations",
        ),
        (
            r"\bapplications of concepts\s+bus structures\b",
            "applications of bus structures",
        ),
        (r"\bbriefly explain\b", "Briefly explain"),
        (r"\banalyze\b", "Analyze"),
    ]
    for pattern, repl in replacements:
        repaired = re.sub(pattern, repl, repaired, flags=re.IGNORECASE)

    repaired = re.sub(r"\s+([,.;:?!])", r"\1", repaired)
    repaired = re.sub(r"\s+", " ", repaired).strip()
    if repaired and repaired[-1] not in ".?!":
        repaired += "."
    if repaired:
        repaired = repaired[0].upper() + repaired[1:]
    return repaired


def _dedupe_key(text):

    base = _repair_question_text(text).lower()
    base = base.rstrip(".?!")
    base = re.sub(
        r"^(define(?: and explain)?|briefly explain|explain|describe|state(?: the)?(?: applications of)?|"
        r"list|name|outline|illustrate(?: the)?(?: applications of)?|differentiate between|"
        r"discuss|analyze|analyse|evaluate|critically evaluate|compare|apply the principles of)\s+",
        "",
        base
    )
    base = re.sub(r"\bwith suitable examples\b", "", base)
    base = re.sub(r"\bin brief\b", "", base)
    base = re.sub(r"\bin detail\b", "", base)
    base = re.sub(r"\bthe key differences between\b", "", base)
    base = re.sub(
        r"^(?:the\s+)?(?:applications?|importance|significance|principle|working principle|"
        r"major causes|causes|effects|impacts|role|procedure|steps)\s+of\s+",
        "",
        base
    )
    tokens = [token for token in re.split(r"[^a-z0-9]+", base) if token]
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


def _extract_question_lines(text):

    questions = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if _is_numbered_question_line(line):
            body = _strip_question_prefix(line)
            if body and not _is_prompt_or_template_line(body):
                if "..." in body:
                    continue
                questions.append(_ensure_tags(body))
    return questions


def _extract_plain_numbered_questions(text):

    questions = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not _is_numbered_question_line(line):
            continue
        body = _strip_question_prefix(line)
        if not body:
            continue
        if _is_prompt_or_template_line(body):
            continue
        if "..." in body:
            continue
        questions.append(_ensure_tags(body))
    return questions


def _extract_questions_grouped_by_section(text):

    grouped = {}
    current_section = "__all__"

    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue

        section_match = re.match(r"^(SECTION\s+[A-Z])\b", line, flags=re.IGNORECASE)
        if section_match:
            current_section = section_match.group(1).upper()
            grouped.setdefault(current_section, [])
            continue

        if not _is_numbered_question_line(line):
            continue

        body = _strip_question_prefix(line)
        if not body:
            continue
        if _is_prompt_or_template_line(body) or "..." in body:
            continue

        grouped.setdefault(current_section, []).append(_ensure_tags(body))

    return grouped


def _is_prompt_or_template_line(line):

    lowered = line.lower()
    blocked_phrases = [
        "generate ",
        "output strictly in this format",
        "for each question",
        "use only these levels",
        "match the target distribution",
        "each question must be",
        "prefer lower-to-mid",
        "prefer higher",
        "mandatory module-wise",
        "modules and extracted concepts",
        "do not repeat module allocation",
        "must start with module and bloom tags",
    ]
    return any(phrase in lowered for phrase in blocked_phrases)


def _is_numbered_question_line(line):
    if not line:
        return False
    return bool(re.match(r"^(?:Q\s*)?\d+\s*[\.\)]\s+", line, flags=re.IGNORECASE))


def _strip_question_prefix(line):
    return re.sub(r"^(?:Q\s*)?\d+\s*[\.\)]\s*", "", line, flags=re.IGNORECASE).strip()


def _build_pattern_preview_lines(paper_model, paper_model_key, module_count):

    lines = ["PATTERN PREVIEW"]
    sections = paper_model.get("sections", [])
    for section in sections:
        lines.append(
            f"{section['name']}: {section['count']} x {section['marks']} = {section['count'] * section['marks']} marks"
        )

    if paper_model_key == "100":
        lines.append("Module Split (Strict):")
        lines.append("SECTION A (3 marks): 2 questions from each Module 1-5")
        lines.append("SECTION B (14 marks): 1 question from each Module 1-5")
    else:
        lines.append("Module Split:")
        lines.append(f"Questions can follow any order across {module_count} uploaded module(s).")

    return lines


def format_paper(
    text,
    title="AI GENERATED QUESTION PAPER",
    total_marks=50,
    duration_hours=2,
    paper_model=None,
    paper_model_key="50",
    module_count=1
):

    lines = []

    lines.append(str(title).strip() or "AI GENERATED QUESTION PAPER")
    lines.append(
        {
            "type": "header_meta",
            "left": f"Duration: {duration_hours} Hours",
            "right": f"Total Marks: {total_marks}",
        }
    )
    lines.append("=" * 50)
    lines.append("")

    if not paper_model or "sections" not in paper_model:
        raise ValueError("paper_model with sections is required for formatting.")

    lines.extend(_build_pattern_preview_lines(paper_model, paper_model_key, module_count))
    lines.append("=" * 50)
    lines.append("")

    grouped_questions = _extract_questions_grouped_by_section(text)
    if not grouped_questions:
        grouped_questions = {"__all__": _extract_plain_numbered_questions(text)}

    cleaned_grouped = {}
    seen_keys = set()
    for section_name, tagged_list in grouped_questions.items():
        cleaned_grouped[section_name] = []
        for tagged in tagged_list:
            display = _to_display_text(tagged)
            key = _dedupe_key(display)
            if key and key in seen_keys:
                continue
            if key:
                seen_keys.add(key)
            module_match = re.search(r"\[Module:\s*([1-5])\]", tagged)
            bloom_match = re.search(
                r"\[Bloom:\s*(Remember|Understand|Apply|Analyze|Evaluate|Create)\]",
                tagged
            )
            module = module_match.group(1) if module_match else "1"
            bloom = bloom_match.group(1) if bloom_match else "Understand"
            cleaned = _repair_question_text(display)
            cleaned_grouped[section_name].append(
                f"[Module: {module}] [Bloom: {bloom}] {cleaned}"
            )

    global_cursor = 0
    global_questions = cleaned_grouped.get("__all__", [])
    for section in paper_model["sections"]:
        lines.append(section["name"])
        section_key = section["name"].upper()
        section_questions = cleaned_grouped.get(section_key, [])
        for idx in range(1, section["count"] + 1):
            if idx - 1 < len(section_questions):
                lines.append(f"{idx}. {_to_display_text(section_questions[idx - 1])}")
            elif global_cursor < len(global_questions):
                lines.append(f"{idx}. {_to_display_text(global_questions[global_cursor])}")
                global_cursor += 1
            else:
                lines.append(
                    f"{idx}. Question generation incomplete. Please regenerate."
                )
        lines.append("")

    return lines
