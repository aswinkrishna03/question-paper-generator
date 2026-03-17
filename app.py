from flask import (
    Flask,
    render_template,
    request,
    send_from_directory,
    redirect,
    url_for,
    session,
    flash,
)
import os
import re
import sqlite3
import random
import time
from werkzeug.security import generate_password_hash, check_password_hash

from concept_extractor import extract_concepts
from question_generator import (
    build_prompt,
    generate_paper,
    generate_paper_gemini,
    generate_paper_fast,
    normalize_blooms_levels,
    normalize_distribution_profile,
    normalize_paper_model,
    get_paper_model,
    get_runtime_device,
    _is_low_quality_question,
    _is_short_answer_style,
    _is_long_answer_style,
    _enforce_short_question_brevity,
)
from formatter import format_paper
from pdf_generator import generate_pdf
from pdf_text_extractor import extract_pdf_content
from gemini_validator import validate_questions_with_gemini, rewrite_questions_with_gemini

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change-this-secret")

UPLOAD_FOLDER = "uploads"
GENERATED_FOLDER = "generated_papers"
DATABASE_PATH = "questions.db"
MIN_VALID_CONCEPTS_PER_MODULE = {
    "50": 2,
    "100": 3,
}
GEMINI_REPAIR_MAX_ATTEMPTS = int(os.environ.get("GEMINI_REPAIR_MAX_ATTEMPTS", "3"))
GEMINI_GENERATION_MAX_ATTEMPTS = int(os.environ.get("GEMINI_GENERATION_MAX_ATTEMPTS", "2"))
GEMINI_MAX_WAIT_SECONDS = int(os.environ.get("GEMINI_MAX_WAIT_SECONDS", "600"))
GEMINI_GENERATION_ENABLED = os.environ.get("GEMINI_GENERATION_ENABLED", "true").lower() in {"1", "true", "yes"}
DEBUG_GENERATION = os.environ.get("DEBUG_GENERATION", "false").lower() in {"1", "true", "yes"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GENERATED_FOLDER, exist_ok=True)


def _gemini_validation_enabled():
    if not os.environ.get("GEMINI_API_KEY", "").strip():
        return False
    flag = os.environ.get("GEMINI_VALIDATION_ENABLED", "true").lower()
    return flag in {"1", "true", "yes"}


def _gemini_generation_enabled():
    if not os.environ.get("GEMINI_API_KEY", "").strip():
        return False
    return GEMINI_GENERATION_ENABLED


def _get_db_connection():

    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def _init_db():

    connection = _get_db_connection()
    cursor = connection.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS examiners (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS generated_paper_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            examiner_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            file_name TEXT NOT NULL,
            total_marks INTEGER NOT NULL,
            duration_hours REAL NOT NULL,
            generated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (examiner_id) REFERENCES examiners(id)
        )
        """
    )
    connection.commit()
    connection.close()


def _current_examiner():

    examiner_id = session.get("examiner_id")
    if not examiner_id:
        return None

    connection = _get_db_connection()
    examiner = connection.execute(
        "SELECT id, username FROM examiners WHERE id = ?",
        (examiner_id,)
    ).fetchone()
    connection.close()
    return examiner


_init_db()


def _count_valid_generated_questions(text):

    count = 0
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not _is_numbered_question_line(line):
            continue
        if "..." in line:
            continue
        if "generate " in line.lower():
            continue
        if "output strictly in this format" in line.lower():
            continue
        count += 1
    return count


def _extract_numbered_question_bodies(text):

    bodies = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not _is_numbered_question_line(line):
            continue
        body = _strip_question_prefix(line)
        if not body:
            continue
        if "..." in body:
            continue
        if "generate " in body.lower():
            continue
        if "output strictly in this format" in body.lower():
            continue
        bodies.append(body)
    return bodies


def _normalize_concept_key(concept):

    key = re.sub(r"\s+", " ", str(concept or "")).strip().lower()
    key = re.sub(r"[^a-z0-9]+", " ", key)
    return re.sub(r"\s+", " ", key).strip()


def _looks_like_valid_concept(concept):

    text = re.sub(r"\s+", " ", str(concept or "")).strip()
    lowered = text.lower()
    if not text:
        return False

    words = text.split()
    if len(words) < 2 or len(words) > 7:
        return False
    question_words = {"what", "when", "where", "which", "who", "whom", "whose", "why", "how"}
    if any(word.lower() in question_words for word in words):
        return False
    if words[-1].lower() in question_words or words[-1].lower() in {"itself", "found", "finding"}:
        return False
    if len(words) >= 3 and len({word.lower() for word in words}) < len(words):
        return False
    if words[0].lower() in {"as", "and", "or"}:
        return False
    if re.search(r"\b(on|of|for|to|in|with|by|and|or)\b$", lowered):
        return False
    if re.search(r"\bas\b$", lowered):
        return False
    if re.search(r"^(of|in|on|to|for|with|by)\b", lowered):
        return False
    if re.search(r"\brequires\s+\d+\s*[vV]\b", lowered):
        return False
    if re.search(r"\b\d+[A-Za-z]{2,}\b", text):
        return False
    if re.search(r"&\s*$", text):
        return False
    if re.search(r"[A-Za-z]{12,}\d+[A-Za-z0-9]*", text):
        return False
    if re.search(r"[A-Z]{4,}[A-Z][a-z]{3,}", text):
        return False
    if re.search(r"[a-z]{20,}", lowered):
        return False
    if re.search(r"[a-z]+[A-Z]{2,}", text):
        return False
    if re.search(r"[A-Z]{2,}[a-z]{3,}", text):
        return False
    if re.search(r"[A-Z]{2,}[A-Z][a-z]{2,}", text):
        return False
    if re.search(r"\b(?:which|that|the|or|and|to|in|on|of|for|with|by)(?:contains|operand|location|register|stands|serial|data|address)\b", lowered):
        return False
    if re.search(
        r"\b(contains|contain|stands for|may be|provide|require|referred using|using the|"
        r"program counter to zero|address of the memory|execution of input or output instructions)\b",
        lowered
    ):
        return False
    if re.search(r"\b(decodes|is computed|iscomputed|computed as|sets the|is applied|after each|restart|as responsible)\b", lowered):
        return False
    if re.search(r"\bas\s+input\s+or\s+as\s+output\s+ports?\b", lowered):
        return False
    if re.search(r"\b(?:\d+(?:\.\d+)?)\s*mhz\b", lowered):
        return False
    if re.search(r"\b\d+\s*h\s+is\b", lowered):
        return False
    if "upward compatible" in lowered:
        return False
    if re.search(r"\bcalled as\b", lowered):
        return False
    clause_markers = {"is", "are", "was", "were", "be", "being", "contains", "contain", "provide", "require", "referred", "using", "may"}
    if len(words) > 4 and any(word.lower() in clause_markers for word in words):
        return False

    blocked_fragments = (
        "real excitement surrounding",
        "programming lab",
        "first out fashion",
        "question paper",
        "course outcome",
        "syllabus",
        "concepts instruction",
        "numbers array",
        "organization horizontal",
        "bus structures memory",
        "transfer arithmetic logic",
        "number unsigned multiplication",
        "dimensional logic array",
        "sequencing addressing modes",
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
    if any(fragment in lowered for fragment in blocked_fragments):
        return False

    return True


def _prepare_module_concepts(raw_concepts, limit=12):

    prepared = []
    seen = set()
    for concept in raw_concepts or []:
        clean = re.sub(r"\s+", " ", str(concept or "")).strip(" .,:;")
        if not _looks_like_valid_concept(clean):
            continue
        key = _normalize_concept_key(clean)
        if not key or key in seen:
            continue
        seen.add(key)
        prepared.append(clean)
        if len(prepared) >= limit:
            break
    return prepared


def _question_uniqueness_key(question_body):

    body = re.sub(r"\[Module:\s*[1-5]\]\s*", "", str(question_body or ""), flags=re.IGNORECASE)
    body = re.sub(
        r"\[Bloom:\s*(Remember|Understand|Apply|Analyze|Evaluate|Create)\]\s*",
        "",
        body,
        flags=re.IGNORECASE
    )
    body = re.sub(r"\s+", " ", body).strip().lower().rstrip(".?!")
    body = re.sub(
        r"^(define(?: and explain)?|briefly explain|explain|describe|state(?: the)?(?: applications of)?|"
        r"list|name|outline|illustrate(?: the)?(?: applications of)?|differentiate between|"
        r"discuss|analyze|analyse|evaluate|critically evaluate|compare|apply the principles of)\s+",
        "",
        body
    )
    body = re.sub(r"\bwith suitable examples\b", "", body)
    body = re.sub(r"\bin brief\b", "", body)
    body = re.sub(r"\bin detail\b", "", body)
    body = re.sub(r"\bthe key differences between\b", "", body)
    body = re.sub(
        r"^(?:the\s+)?(?:applications?|importance|significance|principle|working principle|"
        r"major causes|causes|effects|impacts|role|procedure|steps)\s+of\s+",
        "",
        body
    )
    tokens = [token for token in re.split(r"[^a-z0-9]+", body) if token]
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


def _count_unique_valid_generated_questions(text):

    seen = set()
    for body in _extract_numbered_question_bodies(text):
        key = _question_uniqueness_key(body)
        if not key:
            continue
        seen.add(key)
    return len(seen)

def _numbered_question_line_indices(text):

    lines = text.split("\n")
    indices = []
    for idx, raw in enumerate(lines):
        if _is_numbered_question_line(raw.strip()):
            indices.append(idx)
    return lines, indices


def _is_numbered_question_line(line):
    if not line:
        return False
    return bool(re.match(r"^(?:\*\*|__)?(?:Q\s*)?\d+\s*[\.\)](?:\*\*|__)?\s+", line, flags=re.IGNORECASE))


def _strip_question_prefix(line):
    return re.sub(r"^(?:\*\*|__)?(?:Q\s*)?\d+\s*[\.\)](?:\*\*|__)?\s*", "", line, flags=re.IGNORECASE).strip()


def _merge_rejected_questions(original_text, candidate_text, rejected_indices):

    original_lines, original_line_indices = _numbered_question_line_indices(original_text)
    candidate_bodies = _extract_numbered_question_bodies(candidate_text)

    if not original_line_indices or not candidate_bodies:
        return original_text
    if len(original_line_indices) != len(candidate_bodies):
        return original_text

    for rejected in rejected_indices:
        if rejected <= 0 or rejected > len(original_line_indices):
            continue
        line_idx = original_line_indices[rejected - 1]
        prefix_match = re.match(r"^(\d+\.\s+)", original_lines[line_idx].strip())
        prefix = prefix_match.group(1) if prefix_match else ""
        replacement_body = candidate_bodies[rejected - 1]
        original_lines[line_idx] = f"{prefix}{replacement_body}"

    return "\n".join(original_lines)

def _index_styles_for_paper(paper_model):

    index_styles = {}
    cursor = 0
    for section in paper_model.get("sections", []):
        style = "SHORT" if section.get("marks", 0) <= 3 else "LONG"
        for idx in range(cursor + 1, cursor + section.get("count", 0) + 1):
            index_styles[idx] = style
        cursor += section.get("count", 0)
    return index_styles


def _apply_rewrites(original_text, rewrite_map):

    if not rewrite_map:
        return original_text
    original_lines, original_line_indices = _numbered_question_line_indices(original_text)
    if not original_line_indices:
        return original_text
    for index, question in rewrite_map.items():
        if index <= 0 or index > len(original_line_indices):
            continue
        line_idx = original_line_indices[index - 1]
        prefix_match = re.match(r"^(\d+\.\s+)", original_lines[line_idx].strip())
        prefix = prefix_match.group(1) if prefix_match else ""
        original_lines[line_idx] = f"{prefix}{question}"
    return "\n".join(original_lines)


def _is_valid_rewrite(question_text, style):

    if _is_low_quality_question(question_text):
        return False
    if style == "SHORT":
        adjusted = _enforce_short_question_brevity(question_text, max_words=12)
        if not _is_short_answer_style(adjusted):
            return False
        if len(adjusted.rstrip(".?!").split()) > 12:
            return False
    else:
        if not _is_long_answer_style(question_text):
            return False
    return True


def _repair_questions_with_gemini(
    text,
    concepts,
    module_concepts,
    blooms_levels,
    generator_fn,
    paper_model_key,
    paper_model,
):

    if not _gemini_validation_enabled():
        return text, True

    index_styles = _index_styles_for_paper(paper_model)
    working = text
    regeneration_used = False
    for _ in range(max(1, GEMINI_REPAIR_MAX_ATTEMPTS)):
        questions = _extract_numbered_question_bodies(working)
        if not questions:
            return working, False

        verdict = validate_questions_with_gemini(
            questions=questions,
            concepts=concepts,
            module_concepts=module_concepts,
            blooms_levels=blooms_levels,
        )
        if verdict.get("ok") and not verdict.get("rejected"):
            return working, True

        rejected = verdict.get("rejected") or []
        rejected_indices = [
            int(item.get("index", 0) or 0)
            for item in rejected
            if int(item.get("index", 0) or 0) > 0
        ]
        if not rejected_indices:
            if generator_fn and not regeneration_used:
                candidate = generator_fn()
                if candidate:
                    regeneration_used = True
                    working = candidate
                    continue
            return working, False

        rewrite_result = rewrite_questions_with_gemini(
            questions=questions,
            rejected_indices=rejected_indices,
            index_styles=index_styles,
            concepts=concepts,
            module_concepts=module_concepts,
            blooms_levels=blooms_levels,
        )
        if rewrite_result.get("ok") and rewrite_result.get("rewrites"):
            rewrite_map = {}
            for item in rewrite_result.get("rewrites", []):
                idx = int(item.get("index", 0) or 0)
                question = str(item.get("question", "")).strip()
                style = index_styles.get(idx, "SHORT")
                if idx > 0 and question and _is_valid_rewrite(question, style):
                    rewrite_map[idx] = question
            if rewrite_map:
                working = _apply_rewrites(working, rewrite_map)
                continue

        if generator_fn and not regeneration_used:
            candidate = generator_fn()
            if not candidate:
                return working, False
            regeneration_used = True
            merged = _merge_rejected_questions(working, candidate, rejected_indices)

            if paper_model_key == "100" and not _is_valid_100_module_distribution(merged):
                working = candidate
            else:
                working = merged
            continue

        return working, False

    return working, False


def _has_illogical_question_fragments(text):

    blocked_keys = {
        "non conventional",
        "conventional",
        "high power",
        "low power",
        "clean development",
        "sustainable",
        "development goal",
        "committee",
        "commission",
        "board",
        "council",
        "programming lab",
        "real excitement surrounding",
        "first out fashion",
        "while the",
        "they are trap",
        "in the above example",
        "inthe above example",
    }
    for body in _extract_numbered_question_bodies(text):
        key = _question_uniqueness_key(body)
        if not key:
            continue
        if key in blocked_keys:
            return True
        if re.search(r"\b(real excitement surrounding|programming lab|first out fashion)\b", body, flags=re.IGNORECASE):
            return True
        if re.search(
            r"\b(lines each|three groups of|while the|they are trap|in the above example|inthe above example|"
            r"programmed in three different modes|interfaced together to system bus|address on to the stack)\b",
            body,
            flags=re.IGNORECASE
        ):
            return True
        if re.search(r"\b(on|of|for|to|in|with|by|and|or)\.$", body.strip(), flags=re.IGNORECASE):
            return True
        if re.search(r"\brequires\s+\d+\s*[vV]\b", body, flags=re.IGNORECASE):
            return True
        if re.search(r"&\s*[.?!]?$", body):
            return True
        if re.search(r"[A-Za-z]{12,}\d+[A-Za-z0-9]*", body):
            return True
        if re.search(r"[A-Z]{4,}[A-Z][a-z]{3,}", body):
            return True
        if re.search(r"[a-z]{20,}", body.lower()):
            return True
        if re.search(r"[a-z]+[A-Z]{2,}", body):
            return True
        if re.search(r"[A-Z]{2,}[a-z]{3,}", body):
            return True
        if re.search(r"[A-Z]{2,}[A-Z][a-z]{2,}", body):
            return True
        if re.search(r"\b(?:which|that|the|or|and|to|in|on|of|for|with|by)(?:contains|operand|location|register|stands|serial|data|address)\b", body, flags=re.IGNORECASE):
            return True
        if re.search(
            r"\b(contains one|stands for|may be defined|provide or require|referred using|using the particular|"
            r"program counter to zero|address of the memory ?location|execution of input or output instructions)\b",
            body,
            flags=re.IGNORECASE
        ):
            return True
        if re.search(r"\bcalled as\b", body, flags=re.IGNORECASE):
            return True
        if re.search(
            r"^(define|list|state|explain|briefly explain|discuss|analyze|analyse|evaluate|critically evaluate)\s+(of|in|on|to|for|with|by)\b",
            body.strip(),
            flags=re.IGNORECASE
        ):
            return True
    return False


def _is_valid_100_module_distribution(text):

    questions = _extract_numbered_question_bodies(text)
    if len(questions) < 15:
        return False

    module_pattern = re.compile(r"\[Module:\s*([1-5])\]")

    section_a = questions[:10]
    section_b = questions[10:15]
    if len(section_a) < 10 or len(section_b) < 5:
        return False

    a_counts = {idx: 0 for idx in range(1, 6)}
    b_counts = {idx: 0 for idx in range(1, 6)}

    for q in section_a:
        match = module_pattern.search(q)
        if not match:
            return False
        a_counts[int(match.group(1))] += 1

    for q in section_b:
        match = module_pattern.search(q)
        if not match:
            return False
        b_counts[int(match.group(1))] += 1

    return all(a_counts[idx] == 2 for idx in range(1, 6)) and all(
        b_counts[idx] == 1 for idx in range(1, 6)
    )


def _passes_gemini_validation(text, concepts, module_concepts, blooms_levels, required_questions):
    if not _gemini_validation_enabled():
        return True

    questions = _extract_numbered_question_bodies(text)
    if not questions:
        return False

    verdict = validate_questions_with_gemini(
        questions=questions,
        concepts=concepts,
        module_concepts=module_concepts,
        blooms_levels=blooms_levels,
    )

    if not verdict.get("ok"):
        app.logger.warning(
            "Gemini validation failed: %s (accepted %s/%s).",
            verdict.get("reason"),
            verdict.get("accepted_count"),
            verdict.get("total"),
        )
        return False

    if verdict.get("accepted_count", 0) < required_questions:
        app.logger.warning(
            "Gemini validation accepted fewer questions than required: %s/%s.",
            verdict.get("accepted_count"),
            required_questions,
        )
        return False

    if verdict.get("rejected"):
        app.logger.warning(
            "Gemini rejected %s question(s).",
            len(verdict.get("rejected", [])),
        )
        return False

    return True


@app.route("/", methods=["GET", "POST"])
def upload_pdf():
    examiner = _current_examiner()
    if not examiner:
        return redirect(url_for("login"))

    if request.method == "POST":

        pdf_files = [f for f in request.files.getlist("pdfs") if f and f.filename]
        if not pdf_files:
            legacy_file = request.files.get("pdf")
            if legacy_file and legacy_file.filename:
                pdf_files = [legacy_file]

        if not pdf_files:
            return "Please upload at least one PDF."

        if len(pdf_files) > 5:
            return "Please upload a maximum of 5 PDFs (one per module)."

        selected_levels = request.form.getlist("blooms_levels")
        blooms_levels = normalize_blooms_levels(selected_levels)
        blooms_distribution = normalize_distribution_profile(
            request.form.get("blooms_distribution", "balanced")
        )
        paper_model_key = normalize_paper_model(request.form.get("paper_model", "50"))
        paper_model = get_paper_model(paper_model_key)
        paper_title = request.form.get("paper_title", "").strip()
        if not paper_title:
            paper_title = "AI GENERATED QUESTION PAPER"

        if paper_model_key == "100" and len(pdf_files) != 5:
            return "For 100 marks model, upload exactly 5 module PDFs."

        all_concepts = []
        module_concepts = []
        weak_modules = []
        min_per_module = MIN_VALID_CONCEPTS_PER_MODULE.get(paper_model_key, 2)
        for idx, pdf_file in enumerate(pdf_files, start=1):
            file_path = os.path.join(UPLOAD_FOLDER, pdf_file.filename)
            pdf_file.save(file_path)

            text, parser_used = extract_pdf_content(file_path)
            if parser_used in {"docling", "marker"}:
                app.logger.info(
                    "Extracted structured markdown for %s using %s",
                    pdf_file.filename,
                    parser_used,
                )

            concepts = extract_concepts(text)
            prepared_concepts = _prepare_module_concepts(concepts, limit=12)
            if prepared_concepts:
                all_concepts.extend(prepared_concepts)
            if len(prepared_concepts) < min_per_module:
                weak_modules.append(
                    f"Module {idx} ({len(prepared_concepts)} valid concept(s))"
                )
            module_concepts.append(
                {
                    "module": idx,
                    "filename": pdf_file.filename,
                    "concepts": prepared_concepts,
                }
            )

        if weak_modules:
            return (
                "Insufficient valid concepts extracted from: "
                + ", ".join(weak_modules)
                + f". Need at least {min_per_module} valid concepts per module. "
                "Upload cleaner syllabus/topic PDFs."
            )

        concepts = all_concepts[:60]
        if len(concepts) == 0:
            return "Could not extract concepts from uploaded PDFs."

        prompt = build_prompt(
            concepts,
            blooms_levels,
            blooms_distribution,
            paper_model_key,
            module_concepts=module_concepts
        )

        required_questions = sum(
            section["count"] for section in paper_model["sections"]
        )
        generated_text = ""
        generation_succeeded = False
        best_candidate_text = ""
        best_unique = 0
        best_valid = 0
        best_illogical = True
        best_distribution_ok = False

        def _record_best(candidate_text):
            nonlocal best_candidate_text, best_unique, best_valid, best_illogical, best_distribution_ok
            if not candidate_text:
                return
            if DEBUG_GENERATION:
                try:
                    with open(os.path.join(GENERATED_FOLDER, "last_generation.txt"), "w", encoding="utf-8") as fh:
                        fh.write(candidate_text)
                except Exception:
                    pass
            found = _count_valid_generated_questions(candidate_text)
            unique_found = _count_unique_valid_generated_questions(candidate_text)
            illogical = _has_illogical_question_fragments(candidate_text)
            distribution_ok = True
            if paper_model_key == "100":
                distribution_ok = _is_valid_100_module_distribution(candidate_text)

            if unique_found > best_unique or (
                unique_found == best_unique and found > best_valid
            ):
                best_candidate_text = candidate_text
                best_unique = unique_found
                best_valid = found
                best_illogical = illogical
                best_distribution_ok = distribution_ok
                # already dumped above when DEBUG_GENERATION is enabled

        def _candidate_passes_structure(candidate_text):
            if not candidate_text:
                return False
            found = _count_valid_generated_questions(candidate_text)
            unique_found = _count_unique_valid_generated_questions(candidate_text)
            if found < required_questions or unique_found < required_questions:
                return False
            if _has_illogical_question_fragments(candidate_text):
                return False
            if paper_model_key == "100" and not _is_valid_100_module_distribution(candidate_text):
                return False
            return True

        def _try_finalize_candidate(candidate_text, generator_fn):
            nonlocal generated_text, generation_succeeded
            if not candidate_text:
                return False

            _record_best(candidate_text)
            if not _candidate_passes_structure(candidate_text):
                return False

            repaired_text, gemini_ok = _repair_questions_with_gemini(
                candidate_text,
                concepts,
                module_concepts,
                blooms_levels,
                generator_fn,
                paper_model_key,
                paper_model,
            )
            if not gemini_ok:
                return False

            _record_best(repaired_text)
            if not _candidate_passes_structure(repaired_text):
                return False

            generated_text = repaired_text
            generation_succeeded = True
            return True

        try:
            if _gemini_generation_enabled():
                gemini_generator = lambda: generate_paper_gemini(prompt)
                gemini_attempts = max(1, GEMINI_GENERATION_MAX_ATTEMPTS)
                for _ in range(gemini_attempts):
                    candidate = gemini_generator()
                    if _try_finalize_candidate(candidate, gemini_generator):
                        break

                if not generation_succeeded and _gemini_validation_enabled():
                    wait_deadline = time.monotonic() + max(60, GEMINI_MAX_WAIT_SECONDS)
                    seed_candidate = best_candidate_text
                    while time.monotonic() < wait_deadline and not generation_succeeded:
                        if seed_candidate:
                            candidate = seed_candidate
                            seed_candidate = ""
                        else:
                            candidate = gemini_generator()
                        if _try_finalize_candidate(candidate, gemini_generator):
                            break
            elif get_runtime_device() == "cpu":
                cpu_attempts = 4 if paper_model_key == "100" else 2
                for _ in range(cpu_attempts):
                    trial_concepts = list(concepts)
                    random.shuffle(trial_concepts)
                    generated_text = generate_paper_fast(
                        concepts=trial_concepts,
                        paper_model=paper_model,
                        blooms_levels=blooms_levels,
                        module_concepts=module_concepts,
                        paper_model_key=paper_model_key
                    )
                    _record_best(generated_text)
                    def _regen_cpu():
                        regen_concepts = list(concepts)
                        random.shuffle(regen_concepts)
                        return generate_paper_fast(
                            concepts=regen_concepts,
                            paper_model=paper_model,
                            blooms_levels=blooms_levels,
                            module_concepts=module_concepts,
                            paper_model_key=paper_model_key
                        )
                    unique_found = _count_unique_valid_generated_questions(generated_text)
                    if _has_illogical_question_fragments(generated_text):
                        continue
                    repaired_text, gemini_ok = _repair_questions_with_gemini(
                        generated_text,
                        concepts,
                        module_concepts,
                        blooms_levels,
                        _regen_cpu,
                        paper_model_key,
                        paper_model,
                    )
                    if not gemini_ok:
                        continue
                    repaired_unique = _count_unique_valid_generated_questions(repaired_text)
                    _record_best(repaired_text)
                    if repaired_unique >= required_questions:
                        generated_text = repaired_text
                        generation_succeeded = True
                        break
            else:
                max_attempts = 5 if paper_model_key == "100" else 3
                for _ in range(max_attempts):
                    candidate = generate_paper(prompt)
                    _record_best(candidate)
                    found = _count_valid_generated_questions(candidate)
                    unique_found = _count_unique_valid_generated_questions(candidate)
                    generated_text = candidate
                    if found < required_questions:
                        continue
                    if unique_found < required_questions:
                        continue
                    if _has_illogical_question_fragments(candidate):
                        continue
                    if paper_model_key == "100" and not _is_valid_100_module_distribution(candidate):
                        continue
                    repaired_text, gemini_ok = _repair_questions_with_gemini(
                        candidate,
                        concepts,
                        module_concepts,
                        blooms_levels,
                        lambda: generate_paper(prompt),
                        paper_model_key,
                        paper_model,
                    )
                    if not gemini_ok:
                        continue
                    repaired_unique = _count_unique_valid_generated_questions(repaired_text)
                    _record_best(repaired_text)
                    if found >= required_questions and repaired_unique >= required_questions:
                        generated_text = repaired_text
                        generation_succeeded = True
                        break
        except RuntimeError as api_err:
            if "Rate limit exceeded" in str(api_err):
                return (
                    f"Gemini API Error: {str(api_err)}. "
                    "The Free API has strict usage limits. Please wait a minute and try again."
                )
            raise

        if not generation_succeeded:
            if _gemini_validation_enabled():
                return (
                    "Question generation waited for the Gemini rate-limit window, "
                    "but full validation still did not complete in time. "
                    "Please try again."
                )
            if best_candidate_text:
                app.logger.warning(
                    "Generation pattern not satisfied; using best candidate "
                    "(unique=%s, valid=%s, illogical=%s, distribution_ok=%s).",
                    best_unique,
                    best_valid,
                    best_illogical,
                    best_distribution_ok,
                )
                flash(
                    "Pattern checks did not fully pass, but a best-effort paper was "
                    "generated. You can regenerate for higher quality."
                )
                generated_text = best_candidate_text
            else:
                return (
                    "Question generation did not satisfy the required pattern. "
                    "Please regenerate."
                )

        duration_hours = 1.5 if paper_model["total_marks"] == 50 else 3
        paper_lines = format_paper(
            generated_text,
            title=paper_title,
            total_marks=paper_model["total_marks"],
            duration_hours=duration_hours,
            paper_model=paper_model,
            paper_model_key=paper_model_key,
            module_count=len(module_concepts)
        )

        filename = generate_pdf(paper_lines)

        connection = _get_db_connection()
        connection.execute(
            """
            INSERT INTO generated_paper_history (
                examiner_id,
                title,
                file_name,
                total_marks,
                duration_hours
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                examiner["id"],
                paper_title,
                filename,
                paper_model["total_marks"],
                duration_hours,
            )
        )
        connection.commit()
        connection.close()

        return render_template(
            "result.html",
            file_name=filename,
            paper_title=paper_title
        )

    return render_template("upload.html", examiner_username=examiner["username"])


@app.route("/login", methods=["GET", "POST"])
def login():

    if _current_examiner():
        return redirect(url_for("upload_pdf"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if not username or not password:
            flash("Username and password are required.")
            return render_template("login.html")

        connection = _get_db_connection()
        examiner = connection.execute(
            "SELECT id, username, password_hash FROM examiners WHERE username = ?",
            (username,)
        ).fetchone()
        connection.close()

        if not examiner or not check_password_hash(examiner["password_hash"], password):
            flash("Invalid username or password.")
            return render_template("login.html")

        session["examiner_id"] = examiner["id"]
        session["examiner_username"] = examiner["username"]
        return redirect(url_for("upload_pdf"))

    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():

    if _current_examiner():
        return redirect(url_for("upload_pdf"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if not username or not password:
            flash("Username and password are required.")
            return render_template("register.html")

        connection = _get_db_connection()
        existing = connection.execute(
            "SELECT id FROM examiners WHERE username = ?",
            (username,)
        ).fetchone()
        if existing:
            connection.close()
            flash("Username already exists. Please choose another username.")
            return render_template("register.html")

        connection.execute(
            "INSERT INTO examiners (username, password_hash) VALUES (?, ?)",
            (username, generate_password_hash(password))
        )
        connection.commit()
        connection.close()
        flash("Account created successfully. Please login.")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/logout")
def logout():

    session.clear()
    return redirect(url_for("login"))


@app.route("/history")
def history():

    examiner = _current_examiner()
    if not examiner:
        return redirect(url_for("login"))

    connection = _get_db_connection()
    papers = connection.execute(
        """
        SELECT file_name, title, generated_at, total_marks, duration_hours
        FROM generated_paper_history
        WHERE examiner_id = ?
        ORDER BY generated_at DESC, id DESC
        """,
        (examiner["id"],)
    ).fetchall()
    connection.close()

    return render_template("history.html", papers=papers)


@app.route("/download/<filename>")
def download_file(filename):
    examiner = _current_examiner()
    if not examiner:
        return redirect(url_for("login"))

    connection = _get_db_connection()
    paper = connection.execute(
        """
        SELECT id
        FROM generated_paper_history
        WHERE examiner_id = ? AND file_name = ?
        """,
        (examiner["id"], filename)
    ).fetchone()
    connection.close()
    if not paper:
        return "You are not authorized to download this file.", 403

    return send_from_directory(GENERATED_FOLDER, filename, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
