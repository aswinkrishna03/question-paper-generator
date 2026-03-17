import re

HEADER_NOISE_TOKENS = {
    "module",
    "unit",
    "course",
    "syllabus",
    "semester",
    "scheme",
    "regulation",
    "credits",
    "credit",
    "ktu",
    "university",
    "internal",
    "external",
    "exam",
    "question",
    "paper",
}

STOPWORDS = {
    "a", "an", "the", "of", "to", "in", "on", "for", "with", "by", "and", "or", "as",
    "from", "into", "over", "under", "between", "within", "without", "about", "above",
    "below", "this", "that", "these", "those", "its", "their", "his", "her", "their",
    "such", "some", "any", "each", "every", "both", "either", "neither", "etc",
    "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
    "itself",
    "asi",
    "multipurpose",
    "according",
}

VERB_LIKE = {
    "is", "are", "was", "were", "be", "being", "been",
    "has", "have", "had",
    "do", "does", "did",
    "can", "may", "might", "will", "shall", "should", "could", "would",
    "use", "uses", "using",
    "provide", "provides", "provided",
    "require", "requires", "required",
    "contain", "contains", "contained",
    "refer", "refers", "referred",
    "decode", "decodes", "decoded",
    "compute", "computes", "computed",
    "set", "sets", "setting",
    "apply", "applies", "applied",
    "write", "writes", "writing",
    "read", "reads", "reading",
    "restart", "restarts", "restarted",
    "complete", "completes", "completed",
    "drive", "drives", "driven",
    "contain", "contains", "containing",
    "call", "calls", "called",
    "process", "processes", "processed", "processing",
    "find", "finds", "found", "finding",
}

SHORT_TOKEN_WHITELIST = {
    "cpu", "alu", "rom", "ram", "bus", "io", "dma", "spi", "adc", "dac", "usb", "uart",
    "irq", "inta", "pi", "sr", "psw",
}

DOMAIN_TERMS = {
    "8085 microprocessor",
    "8086 microprocessor",
    "8255 programmable peripheral interface",
    "programmable peripheral interface",
    "programmable interrupt controller",
    "interrupt service routine",
    "interrupt request",
    "interrupt acknowledge",
    "interrupt programming",
    "interrupt flag instruction",
    "register indirect",
    "register relative",
    "based indexed",
    "relative based indexed",
    "intrasegment direct mode",
    "intrasegment indirect mode",
    "intersegment direct",
    "intersegment indirect",
    "stack pointer",
    "data bus buffer",
    "data bus",
    "read/write control logic",
    "control words",
    "maximum mode",
    "minimum mode",
    "i/o read",
    "i/o write",
    "effective address",
    "input output port",
    "serial interface",
    "interrupt control",
    "data memory",
    "special function registers",
    "bus control",
}

GENERIC_TOKENS = {
    "data", "control", "interface", "port", "ports", "interrupt", "serial", "mode",
    "words", "buffer", "bus", "register", "memory", "stack", "address", "instruction",
    "acknowledge", "request", "service", "routine", "peripheral", "programmable",
    "controller", "input", "output", "clock", "single", "integrated", "circuit",
}


def clean_text(text):

    text = text.replace("\r", "\n")
    text = text.replace("â– ", " ")
    text = text.replace("•", " ")
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text


def _normalize_candidate(raw):

    line = raw.strip(" -:;,.")
    line = re.sub(r"\b[A-Z]{2,10}\s*[-]?\s*\d{2,3}\b", " ", line)
    line = re.sub(r"\b[A-Z]{2,5}\s*[-]?\s*\d{2,3}\b", " ", line)
    line = re.sub(r"^(?:module|unit|m)\s*\d+\s*[-:]\s*", "", line, flags=re.IGNORECASE)
    line = re.sub(r"^\d+\s*[-:]\s*", "", line)
    line = re.sub(r"\b(?:module|unit)\s*[-:]?\s*\d+\b", " ", line, flags=re.IGNORECASE)
    line = re.sub(r"^[A-Z][A-Z\s]{3,}\s*-\s*", "", line)
    line = re.sub(r"^\(?[a-zA-Z0-9ivxIVX]{1,4}[\).:-]\s*", "", line)
    line = re.sub(r"^[a-zA-Z]\s+", "", line)
    line = re.sub(r"([a-z])([A-Z])", r"\1 \2", line)
    line = re.sub(r"([A-Za-z])(\d)", r"\1 \2", line)
    line = re.sub(r"(\d)([A-Za-z])", r"\1 \2", line)
    line = re.sub(r"\beffectiveaddress\b", "effective address", line, flags=re.IGNORECASE)
    line = re.sub(r"\bin(this|that|the|case)\b", r"in \1", line, flags=re.IGNORECASE)
    line = re.sub(r"\bbasedindexed\b", "based indexed", line, flags=re.IGNORECASE)
    line = re.sub(r"\banimmediate\b", "an immediate", line, flags=re.IGNORECASE)
    line = re.sub(r"\band\s*([ivx]{1,4})\b", "", line, flags=re.IGNORECASE)
    line = re.sub(r"\band[ivx]{1,4}\b", "", line, flags=re.IGNORECASE)
    line = re.sub(r"\bmodeb\b", "mode b", line, flags=re.IGNORECASE)
    line = re.sub(r"\b\d+\b$", "", line)
    line = re.sub(r"\b(a|an)(single|integrated|output|input|interrupt|instruction|address|microprocessor|controller|interface|register|memory|bus|port)\b", r"\1 \2", line, flags=re.IGNORECASE)
    line = re.sub(r"\band(provides|requires|contains|gives|shows|stores|fetches)\b", r"and \1", line, flags=re.IGNORECASE)
    line = re.sub(r"[_|]+", " ", line)
    line = re.sub(r"[^A-Za-z0-9 ,/&-]", " ", line)
    line = re.sub(r"\s+", " ", line).strip()
    return line


def _split_candidate(candidate):

    parts = [candidate]
    comma_parts = [part.strip(" -") for part in candidate.split(",") if part.strip(" -")]
    if len(comma_parts) >= 2:
        short_parts = [part for part in comma_parts if 1 <= len(part.split()) <= 5]
        if len(short_parts) == len(comma_parts):
            parts = short_parts

    expanded = []
    for part in parts:
        if " - " in part:
            hyphen_parts = [item.strip(" -") for item in part.split(" - ") if item.strip(" -")]
            short_hyphen_parts = [item for item in hyphen_parts if 1 <= len(item.split()) <= 6]
            if len(short_hyphen_parts) >= 2:
                expanded.extend(short_hyphen_parts)
                continue

        if " and " in part.lower():
            and_parts = [item.strip(" -") for item in re.split(r"\band\b", part, flags=re.IGNORECASE) if item.strip(" -")]
            short_parts = [item for item in and_parts if 1 <= len(item.split()) <= 4]
            if len(short_parts) == len(and_parts) and len(short_parts) > 1:
                expanded.extend(short_parts)
                continue
        expanded.append(part)

    refined = []
    for part in expanded:
        words = part.split()
        if len(words) >= 6:
            chunks = []
            current = [words[0]]
            for token in words[1:]:
                if token[0].isupper() and not token.isupper() and len(current) >= 2:
                    chunks.append(" ".join(current))
                    current = [token]
                else:
                    current.append(token)
            if current:
                chunks.append(" ".join(current))
            short_chunks = [chunk.strip(" -") for chunk in chunks if 2 <= len(chunk.split()) <= 5]
            if len(short_chunks) >= 2:
                refined.extend(short_chunks)
                continue
        refined.append(part)

    return refined


def _strip_instructional_prefix(text):

    updated = text.strip()
    patterns = [
        r"^define\s+list\s+of\s+",
        r"^definition\s+of\s+",
        r"^key\s+features\s+of\s+",
        r"^list\s+of\s+",
        r"^define\s+",
        r"^briefly\s+explain\s+",
        r"^explain\s+",
        r"^discuss\s+",
        r"^describe\s+",
        r"^state\s+",
        r"^concepts\s+of\s+",
        r"^fundamental\s+concepts\s+of\s+",
        r"^concepts\s+",
        r"^illustrate\s+practical\s+applications\s+of\s+",
        r"^analyze\s+",
        r"^analyse\s+",
        r"^critically\s+evaluate\s+",
        r"^propose\s+a\s+suitable\s+strategy\s+for\s+",
        r"^design\s+and\s+justify\s+an\s+approach\s+for\s+",
        r"^apply\s+the\s+principles\s+of\s+",
    ]
    for pattern in patterns:
        updated = re.sub(pattern, "", updated, flags=re.IGNORECASE)

    updated = re.sub(
        r"\s+(in\s+brief|in\s+detail|with\s+suitable\s+examples|to\s+an\s+appropriate\s+case)\.?$",
        "",
        updated,
        flags=re.IGNORECASE
    )
    updated = re.sub(
        r"\b(in\s+brief|in\s+detail|with\s+suitable\s+examples|to\s+an\s+appropriate\s+case)\b",
        "",
        updated,
        flags=re.IGNORECASE
    )
    updated = re.sub(r"^(?:the|a|an)\s+", "", updated, flags=re.IGNORECASE)
    updated = re.sub(r"\s+", " ", updated).strip(" -:;,.")
    return updated.strip(" -:;,.")


def _tokenize_words(line):

    return [token for token in re.split(r"[^A-Za-z0-9]+", line) if token]


def _is_valid_token(token):

    low = token.lower()
    if not low:
        return False
    if low in STOPWORDS or low in VERB_LIKE:
        return False
    if low in HEADER_NOISE_TOKENS:
        return False
    if len(low) == 1 and not low.isdigit():
        return False
    if len(low) <= 2 and not low.isdigit() and low not in SHORT_TOKEN_WHITELIST:
        return False
    if len(low) == 3 and low not in SHORT_TOKEN_WHITELIST and not low.isdigit():
        return False
    if re.match(r"^\d+$", low):
        return 2 <= len(low) <= 4
    if re.search(r"\d+[a-z]{2,}", low):
        return False
    if not re.search(r"[a-z]", low):
        return False
    return len(low) >= 2


def _phrase_score(phrase, tokens):

    lowered = phrase.lower()
    if lowered in DOMAIN_TERMS:
        return 10 + len(tokens)
    score = len(tokens)
    digit_tokens = sum(1 for t in tokens if t.isdigit())
    if digit_tokens >= 2:
        score -= 2
    if any(t.lower() in VERB_LIKE for t in tokens):
        score -= 3
    if any(t.lower() in STOPWORDS for t in tokens):
        score -= 2
    if lowered.endswith("as"):
        score -= 3
    return score


def _extract_keyphrases(text, limit=12):

    candidates = {}
    order = []

    for raw_line in text.split("\n"):
        line = _normalize_candidate(_strip_instructional_prefix(raw_line))
        if not line:
            continue
        tokens = _tokenize_words(line)
        if not tokens:
            continue

        for start in range(len(tokens)):
            if not _is_valid_token(tokens[start]):
                continue
            for end in range(start + 1, min(len(tokens), start + 4) + 1):
                window = tokens[start:end]
                if not all(_is_valid_token(tok) for tok in window):
                    break
                phrase = " ".join(window)
                if len(window) >= 4 and phrase.lower() not in DOMAIN_TERMS:
                    continue
                if _is_noise(phrase) or not _looks_natural_phrase(phrase):
                    continue
                score = _phrase_score(phrase, window)
                if score < 2:
                    continue
                key = phrase.lower()
                if key not in candidates:
                    candidates[key] = (phrase, score)
                    order.append(key)
                else:
                    if score > candidates[key][1]:
                        candidates[key] = (phrase, score)

    scored = [(candidates[key][0], candidates[key][1], idx) for idx, key in enumerate(order)]
    scored.sort(key=lambda item: (-item[1], len(item[0].split()), item[2]))
    return [item[0] for item in scored[:limit]]


def _is_noise(line):

    lowered = line.lower()
    blocked_patterns = [
        r"downloaded from",
        r"ktunotes",
        r"page \d+",
        r"copyright",
        r"syllabus",
        r"course outcomes?",
        r"module \d+",
        r"unit \d+",
        r"question bank",
        r"section [ab]",
        r"www\.",
        r"^\d+$",
    ]
    if any(re.search(pattern, lowered) for pattern in blocked_patterns):
        return True

    if re.search(r"\b[A-Z]{2,}\d{2,}\b", line):
        return True
    if re.search(r"\b[A-Z]{2,10}\s*\d{2,3}\b", line):
        return True
    if re.search(r"\b\d+[A-Za-z]{2,}\b", line):
        return True
    if re.search(r"\brequires\s+\d+\s*[vV]\b", lowered):
        return True
    if re.search(r"&\s*$", line):
        return True
    if re.search(r"\b(?:\d+(?:\.\d+)?)\s*mhz\b", lowered):
        return True
    if re.search(r"\b(decodes|computed|iscomputed)\b", lowered):
        return True
    if re.search(r"\bcontaining\b", lowered):
        return True
    if re.search(r"\bcalled\b", lowered):
        return True
    if re.search(r"\baccording\b", lowered):
        return True
    if re.search(r"\b\d+\s*h\s+is\b", lowered):
        return True
    if re.search(r"[A-Za-z]{12,}\d+[A-Za-z0-9]*", line):
        return True
    if re.search(r"[A-Z]{4,}[A-Z][a-z]{3,}", line):
        return True
    if re.search(r"[a-z]{20,}", lowered):
        return True
    if re.search(r"[a-z]+[A-Z]{2,}", line):
        return True
    if re.search(r"[A-Z]{2,}[a-z]{3,}", line):
        return True
    if re.search(r"[A-Z]{2,}[A-Z][a-z]{2,}", line):
        return True
    if re.search(r"\b(?:which|that|the|or|and|to|in|on|of|for|with|by)(?:contains|operand|location|register|stands|serial|data|address)\b", lowered):
        return True
    if re.search(r"\b(lines each|three groups of|while the|they are trap|inthe above example|in the above example)\b", lowered):
        return True
    if re.search(r"\b(?:instruction|data)\s+register\s*&", lowered):
        return True
    if re.search(
        r"\b(contains one|stands for|may be defined|provide or require|referred using|using the particular|"
        r"program counter to zero|address of the memory ?location|execution of input or output instructions)\b",
        lowered
    ):
        return True
    if re.search(r"\b(decodes|is computed|iscomputed|computed as|sets the|is applied|after each|using|restart)\b", lowered):
        return True
    if re.search(r"\bfirst completes\b", lowered):
        return True

    words = line.split()
    if len(words) > 12:
        return True
    if len(words) < 2:
        return True

    lowered_words = [word.lower() for word in words]
    noise_hits = sum(word in HEADER_NOISE_TOKENS for word in lowered_words)
    if noise_hits >= 1:
        return True

    weak_starts = {
        "its", "their", "these", "those", "this", "that", "such", "as",
        "a", "an", "could", "similarly", "to",
        "briefly", "explain", "discuss", "describe", "state"
    }
    if words and words[0].lower() in weak_starts:
        return True

    weak_phrases = [
        "basic concepts of",
        "general idea about",
        "credit course",
        "course code",
        "course outcome",
        "course outcomes",
        "key features",
        "definition of",
        "object oriented programming using java",
        "object oriented programming",
        "basic structure of computers",
        "architecture module",
        "its source",
        "its sources",
        "its effect",
        "its effects",
        "its utilisation",
        "its utilization",
        "as responsible",
        "above you wants from you",
    ]
    if any(phrase in lowered for phrase in weak_phrases):
        return True

    if len(words) == 1:
        token = words[0]
        allowed_single = {
            "ozone",
            "lead",
            "sustainability",
            "biodiversity",
            "ecosystem",
            "pollution",
            "disaster",
            "mitigation",
            "adaptation",
        }
        weak_single = {"i", "ii", "iii", "iv", "v", "activities", "paining", "protection"}
        low = token.lower()
        if low in weak_single:
            return True
        if low not in allowed_single:
            return True

    alpha_chars = sum(ch.isalpha() for ch in line)
    alpha_ratio = alpha_chars / max(1, len(line))
    if alpha_ratio < 0.65:
        return True

    digit_tokens = sum(any(ch.isdigit() for ch in w) for w in words)
    if digit_tokens > 1:
        return True

    alpha_tokens = [word for word in words if any(ch.isalpha() for ch in word)]
    if alpha_tokens:
        upper_tokens = sum(1 for word in alpha_tokens if word.isupper())
        if len(alpha_tokens) >= 3 and (upper_tokens / len(alpha_tokens)) > 0.6:
            return True

    if re.search(r"\b(i/o|io)\b", lowered):
        if re.search(r"\bi\/?o\s+\w+\s+n\s+i\/?o\b", lowered):
            return True
    if re.search(r"\bi\/?o\s+device\s+\d+\b", lowered):
        return True
    if re.search(r"\bas\s+input\s+or\s+as\s+output\s+ports?\b", lowered):
        return True
    if re.search(r"\b(?:\d+(?:\.\d+)?)\s*mhz\b", lowered):
        return True
    if "upward compatible" in lowered:
        return True
    if re.search(r"\b\d+\s+his\b", lowered):
        return True
    if re.search(r"\b(similarly to a|could be a)\b", lowered):
        return True
    if re.search(r"\bas responsible\b", lowered):
        return True
    if re.search(r"\babove you wants from you\b", lowered):
        return True
    tokens = [tok for tok in re.split(r"[^A-Za-z]+", lowered) if tok]
    if tokens and all(tok in GENERIC_TOKENS for tok in tokens) and lowered not in DOMAIN_TERMS:
        return True

    return False


def _looks_natural_phrase(line):

    words = line.split()
    if not words:
        return False

    if len(line) > 70:
        return False
    if " - " in line and len(words) > 6:
        return False

    bad_starts = {
        "and", "or", "but", "so", "then", "in", "on", "at", "for", "from", "with", "by",
        "been", "being", "has", "have", "had", "is", "are", "was", "were", "due", "as",
        "its", "their", "these", "those", "this", "that", "after", "each",
        "a", "an", "could", "similarly", "to",
        "briefly", "explain", "discuss", "describe", "state",
        "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
    }
    if words[0].lower() in bad_starts:
        return False
    fragment_starts = {"enough", "cause", "causes", "causing", "absorb", "absorbs", "absorbing"}
    if words[0].lower() in fragment_starts:
        return False

    bad_endings = {
        "and", "or", "of", "in", "to", "with", "for", "from", "between", "by",
        "has", "have", "had", "is", "are", "was", "were", "been", "compatible", "as",
        "what", "when", "where", "which", "who", "whom", "whose", "why", "how", "itself",
    }
    if words[-1].lower() in bad_endings:
        return False

    lowered = line.lower()
    if "etc" in lowered:
        return False
    if re.search(r"\brequires\s+\d+\s*[vV]\b", lowered):
        return False
    if re.search(r"&\s*$", line):
        return False
    if re.search(r"\b(?:\d+(?:\.\d+)?)\s*mhz\b", lowered):
        return False
    if re.search(r"\b\d+[A-Za-z]{2,}\b", line):
        return False
    if re.search(r"[A-Za-z]{12,}\d+[A-Za-z0-9]*", line):
        return False
    if re.search(r"[A-Z]{4,}[A-Z][a-z]{3,}", line):
        return False
    if re.search(r"[a-z]{20,}", lowered):
        return False
    if re.search(r"[a-z]+[A-Z]{2,}", line):
        return False
    if re.search(r"[A-Z]{2,}[a-z]{3,}", line):
        return False
    if re.search(r"[A-Z]{2,}[A-Z][a-z]{2,}", line):
        return False
    if any(len(word) >= 16 for word in words):
        return False
    if len(words) == 3 and words[0].istitle() and words[1].istitle():
        if len(words[1]) <= 3 and len(words[2]) <= 4:
            return False
    if re.search(r"\b(?:which|that|the|or|and|to|in|on|of|for|with|by)(?:contains|operand|location|register|stands|serial|data|address)\b", lowered):
        return False
    if re.search(r"\b(lines each|three groups of|while the|they are trap|inthe above example|in the above example)\b", lowered):
        return False
    if re.search(
        r"\b(contains one|stands for|may be defined|provide or require|referred using|using the particular|"
        r"program counter to zero|address of the memory ?location|execution of input or output instructions)\b",
        lowered
    ):
        return False
    if re.search(r"\bas\s+input\s+or\s+as\s+output\s+ports?\b", lowered):
        return False
    if re.search(r"\b(?:\d+(?:\.\d+)?)\s*mhz\s+clock\b", lowered):
        return False
    if "upward compatible" in lowered:
        return False
    fragment_patterns = [
        " to cause ",
        " causing ",
        " from the blood ",
        " weather changes",
        " has been ",
        " been added ",
        " due to ",
        " basic concepts of ",
        " general idea about ",
    ]
    if any(pattern in lowered for pattern in fragment_patterns):
        return False

    if line.count(",") > 2:
        return False

    lowered_words = [word.lower() for word in words]
    if len(lowered_words) >= 3:
        if len(set(lowered_words)) < len(lowered_words):
            return False

    clause_words = {"which", "that", "it", "include", "includes", "including", "thereby"}
    if any(w.lower() in clause_words for w in words):
        return False

    # Reject shouty headings like "ENVIRONMENTAL MANAGEMENT STANDARDS"
    alpha_tokens = [w for w in words if any(ch.isalpha() for ch in w)]
    if alpha_tokens:
        upper_tokens = sum(1 for w in alpha_tokens if w.isupper())
        if upper_tokens / len(alpha_tokens) > 0.7:
            return False

    # Prefer concept-like noun phrases over long clause-like fragments.
    if len(words) > 2:
        verb_like = {
            "is", "are", "was", "were", "means", "has", "have", "can", "will",
            "contains", "contain", "provide", "require", "referred", "using", "may",
            "decode", "decodes", "computed", "iscomputed", "sets", "applied",
            "disable", "disables", "write", "writes", "writing", "restart", "completes"
        }
        if any(w.lower() in verb_like for w in words):
            return False

    return True


def extract_concepts(text):

    text = clean_text(text)
    keyphrases = _extract_keyphrases(text, limit=12)
    if keyphrases:
        return keyphrases

    # Fallback to the legacy splitter if keyphrase extraction returns nothing.
    raw_chunks = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        lowered_line = line.lower()
        if any(token in lowered_line for token in ("syllabus", "course code", "credits", "question paper")):
            continue
        raw_chunks.extend(re.split(r"[.!?;:]", line))

    concepts = []
    seen = set()
    for raw in raw_chunks:
        candidate = _normalize_candidate(raw)
        if not candidate:
            continue

        for part in _split_candidate(candidate):
            clean = _normalize_candidate(_strip_instructional_prefix(part))
            if not clean or _is_noise(clean) or not _looks_natural_phrase(clean):
                continue

            key = clean.lower()
            if key in seen:
                continue
            seen.add(key)
            concepts.append(clean)
            if len(concepts) >= 12:
                return concepts

    return concepts
