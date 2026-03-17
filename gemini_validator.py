import json
import os
import threading
import time

from google import genai


DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
DEFAULT_GEN_MODEL = os.environ.get("GEMINI_GEN_MODEL", DEFAULT_MODEL)
DEFAULT_GEN_TEMPERATURE = float(os.environ.get("GEMINI_GEN_TEMPERATURE", "0.6"))
DEFAULT_GEN_MAX_TOKENS = int(os.environ.get("GEMINI_GEN_MAX_TOKENS", "4096"))
DEFAULT_MAX_QUESTIONS = int(os.environ.get("GEMINI_VALIDATE_MAX_QUESTIONS", "20"))
DEFAULT_MIN_ACCEPT_RATIO = float(os.environ.get("GEMINI_VALIDATE_MIN_ACCEPT_RATIO", "0.7"))
DEFAULT_MAX_REJECTS = int(os.environ.get("GEMINI_VALIDATE_MAX_REJECTS", "3"))
DEFAULT_FAIL_OPEN = os.environ.get("GEMINI_VALIDATE_FAIL_OPEN", "true").lower() in {"1", "true", "yes"}
DEFAULT_REQUESTS_PER_MINUTE = max(1, int(os.environ.get("GEMINI_REQUESTS_PER_MINUTE", "15")))

_RATE_LIMIT_LOCK = threading.Lock()
_REQUEST_TIMESTAMPS = []


def _parse_json_response(text):
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return json.loads(text.strip() or "[]")


def _wait_for_rate_limit_slot():
    window_seconds = 60.0

    while True:
        wait_time = 0.0
        with _RATE_LIMIT_LOCK:
            now = time.monotonic()
            cutoff = now - window_seconds
            while _REQUEST_TIMESTAMPS and _REQUEST_TIMESTAMPS[0] <= cutoff:
                _REQUEST_TIMESTAMPS.pop(0)

            if len(_REQUEST_TIMESTAMPS) < DEFAULT_REQUESTS_PER_MINUTE:
                _REQUEST_TIMESTAMPS.append(now)
                return

            wait_time = max(0.05, window_seconds - (now - _REQUEST_TIMESTAMPS[0]))

        print(
            f"[Gemini] Waiting {wait_time:.1f}s to stay within "
            f"{DEFAULT_REQUESTS_PER_MINUTE} requests/minute."
        )
        time.sleep(wait_time)


def _generate_with_retry(client, model_name, contents, config, max_retries=3):
    last_exc = None
    for attempt in range(max_retries):
        try:
            _wait_for_rate_limit_slot()
            return client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
        except Exception as exc:
            exc_str = str(exc).lower()
            if "429" in exc_str or "quota" in exc_str or "rate limit" in exc_str or "exhausted" in exc_str:
                last_exc = exc
                if attempt < max_retries - 1:
                    delay = 10.0 * (2 ** attempt)
                    print(f"[Gemini] Rate limit hit. Retrying in {delay}s (Attempt {attempt+1}/{max_retries})...")
                    time.sleep(delay)
                continue
            raise
    raise RuntimeError(f"Rate limit exceeded after {max_retries} retries. Last error: {last_exc}")


def _get_client():
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


def _build_prompt(questions, concepts, module_concepts, blooms_levels, start_index=1):
    concepts = concepts or []
    module_concepts = module_concepts or []
    blooms_levels = blooms_levels or []

    module_lines = []
    for module in module_concepts:
        module_num = module.get("module")
        module_topics = module.get("concepts") or []
        if not module_num:
            continue
        module_lines.append(f"Module {module_num}: {', '.join(module_topics) if module_topics else 'No concepts'}")

    prompt = (
        "You are a strict university exam question validator.\n"
        "Validate each question for:\n"
        "- Relevance to the provided concepts/modules\n"
        "- Clarity and correctness (no fragments, no corrupted text)\n"
        "- Academic appropriateness for a question paper\n\n"
        "Reject if the question is irrelevant, malformed, ambiguous, or contains broken/errored phrasing.\n\n"
        f"Bloom levels used: {', '.join(blooms_levels) if blooms_levels else 'Not specified'}\n"
        f"Core concepts (global): {', '.join(concepts) if concepts else 'None provided'}\n"
        "Module concepts:\n"
        f"{chr(10).join(module_lines) if module_lines else 'None provided'}\n\n"
        "Questions to validate (index: text):\n"
        + "\n".join(
            [
                f"{idx}: {text}"
                for idx, text in enumerate(questions, start=start_index)
            ]
        )
    )
    return prompt


def _build_rewrite_prompt(questions, rejected_indices, index_styles, concepts, module_concepts, blooms_levels):
    concepts = concepts or []
    module_concepts = module_concepts or []
    blooms_levels = blooms_levels or []

    module_lines = []
    for module in module_concepts:
        module_num = module.get("module")
        module_topics = module.get("concepts") or []
        if not module_num:
            continue
        module_lines.append(f"Module {module_num}: {', '.join(module_topics) if module_topics else 'No concepts'}")

    style_lines = []
    for idx in rejected_indices:
        style = index_styles.get(idx, "SHORT")
        if style == "SHORT":
            style_lines.append(f"{idx}: SHORT (one sentence, <=12 words, simple stem like Define/Explain/State/List)")
        else:
            style_lines.append(f"{idx}: LONG (analytical, with suitable examples)")

    prompt = (
        "You are a strict university exam question editor.\n"
        "Rewrite ONLY the rejected questions listed below.\n"
        "Each rewrite must be clear, grammatical, academic, and relevant to the concepts.\n"
        "Do NOT include module or Bloom tags.\n\n"
        f"Bloom levels used: {', '.join(blooms_levels) if blooms_levels else 'Not specified'}\n"
        f"Core concepts (global): {', '.join(concepts) if concepts else 'None provided'}\n"
        "Module concepts:\n"
        f"{chr(10).join(module_lines) if module_lines else 'None provided'}\n\n"
        "Rewrite rules by index:\n"
        + "\n".join(style_lines)
        + "\n\n"
        "All questions (index: text):\n"
        + "\n".join([f"{idx}: {text}" for idx, text in enumerate(questions, start=1)])
        + "\n\n"
        "Return JSON as an array of objects: "
        "[{\"index\": <number>, \"question\": \"<rewritten question>\"}] "
        "Only include the rejected indices."
    )
    return prompt


def validate_questions_with_gemini(
    questions,
    concepts=None,
    module_concepts=None,
    blooms_levels=None,
    model_name=None,
    max_questions=None,
    min_accept_ratio=None,
    max_rejects=None,
    fail_open=None,
):
    client = _get_client()
    if not client:
        return {
            "ok": True,
            "reason": "GEMINI_API_KEY not set; validation skipped.",
            "rejected": [],
            "accepted_count": len(questions or []),
            "total": len(questions or []),
            "details": [],
        }

    questions = questions or []
    model_name = model_name or DEFAULT_MODEL
    max_questions = max_questions or DEFAULT_MAX_QUESTIONS
    min_accept_ratio = DEFAULT_MIN_ACCEPT_RATIO if min_accept_ratio is None else min_accept_ratio
    max_rejects = DEFAULT_MAX_REJECTS if max_rejects is None else max_rejects
    fail_open = DEFAULT_FAIL_OPEN if fail_open is None else fail_open

    response_schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "index": {"type": "integer"},
                "verdict": {"type": "string"},
                "relevance": {"type": "number"},
                "reason": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["index", "verdict", "relevance", "reason"],
        },
    }

    rejected = []
    details = []
    accepted_count = 0
    total = len(questions)

    try:
        for start in range(0, len(questions), max_questions):
            sliced_questions = questions[start:start + max_questions]
            prompt = _build_prompt(
                sliced_questions,
                concepts,
                module_concepts,
                blooms_levels,
                start_index=start + 1,
            )
            response = _generate_with_retry(
                client=client,
                model_name=model_name,
                contents=prompt,
                config={
                    "temperature": 0.0,
                    "response_mime_type": "application/json",
                    "response_json_schema": response_schema,
                },
            )
            payload = _parse_json_response(response.text or "")

            batch_rejected = []
            for item in payload if isinstance(payload, list) else []:
                index = int(item.get("index", 0) or 0)
                verdict = str(item.get("verdict", "")).strip().lower()
                relevance = float(item.get("relevance", 0.0) or 0.0)
                reason = str(item.get("reason", "")).strip()
                tags = item.get("tags") if isinstance(item.get("tags"), list) else []

                entry = {
                    "index": index,
                    "verdict": verdict,
                    "relevance": relevance,
                    "reason": reason,
                    "tags": tags,
                }
                details.append(entry)
                if verdict not in {"accept", "ok", "pass"}:
                    batch_rejected.append(entry)

            rejected.extend(batch_rejected)
            accepted_count += max(0, len(sliced_questions) - len(batch_rejected))
    except RuntimeError as rerr:
        raise rerr
    except Exception as exc:
        if fail_open:
            return {
                "ok": True,
                "reason": f"Gemini validation failed; skipped. Error: {exc}",
                "rejected": [],
                "accepted_count": len(questions),
                "total": len(questions),
                "details": [],
            }
        return {
            "ok": False,
            "reason": f"Gemini validation failed: {exc}",
            "rejected": [],
            "accepted_count": 0,
            "total": len(questions),
            "details": [],
        }

    accept_ratio = accepted_count / total if total else 1.0

    ok = accept_ratio >= min_accept_ratio and len(rejected) <= max_rejects

    return {
        "ok": ok,
        "reason": "Gemini validation completed.",
        "rejected": rejected,
        "accepted_count": accepted_count,
        "total": total,
        "details": details,
    }


def rewrite_questions_with_gemini(
    questions,
    rejected_indices,
    index_styles,
    concepts=None,
    module_concepts=None,
    blooms_levels=None,
    model_name=None,
):
    client = _get_client()
    if not client:
        return {"ok": False, "reason": "GEMINI_API_KEY not set; rewrite skipped.", "rewrites": []}

    questions = questions or []
    rejected_indices = [int(idx) for idx in rejected_indices or [] if int(idx) > 0]
    if not rejected_indices:
        return {"ok": True, "reason": "No rejected indices.", "rewrites": []}

    model_name = model_name or DEFAULT_MODEL
    prompt = _build_rewrite_prompt(
        questions, rejected_indices, index_styles or {}, concepts, module_concepts, blooms_levels
    )

    response_schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "index": {"type": "integer"},
                "question": {"type": "string"},
            },
            "required": ["index", "question"],
        },
    }

    try:
        response = _generate_with_retry(
            client=client,
            model_name=model_name,
            contents=prompt,
            config={
                "temperature": 0.2,
                "response_mime_type": "application/json",
                "response_json_schema": response_schema,
            },
        )
        payload = _parse_json_response(response.text or "")
    except RuntimeError as rerr:
        raise rerr
    except Exception as exc:
        return {"ok": False, "reason": f"Gemini rewrite failed: {exc}", "rewrites": []}

    rewrites = []
    for item in payload if isinstance(payload, list) else []:
        idx = int(item.get("index", 0) or 0)
        text = str(item.get("question", "")).strip()
        if idx <= 0 or not text:
            continue
        rewrites.append({"index": idx, "question": text})

    return {"ok": True, "reason": "Gemini rewrite completed.", "rewrites": rewrites}


def generate_questions_with_gemini(
    prompt,
    model_name=None,
    temperature=None,
    max_tokens=None,
):
    client = _get_client()
    if not client:
        return {"ok": False, "reason": "GEMINI_API_KEY not set; generation skipped.", "text": ""}

    model_name = model_name or DEFAULT_GEN_MODEL
    temperature = DEFAULT_GEN_TEMPERATURE if temperature is None else temperature
    max_tokens = DEFAULT_GEN_MAX_TOKENS if max_tokens is None else max_tokens

    try:
        response = _generate_with_retry(
            client=client,
            model_name=model_name,
            contents=str(prompt or ""),
            config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            },
        )
        text = (response.text or "").strip()
        if not text:
            return {"ok": False, "reason": "Gemini returned empty response.", "text": ""}
        return {"ok": True, "reason": "Gemini generation completed.", "text": text}
    except RuntimeError as rerr:
        raise rerr
    except Exception as exc:
        return {"ok": False, "reason": f"Gemini generation failed: {exc}", "text": ""}
