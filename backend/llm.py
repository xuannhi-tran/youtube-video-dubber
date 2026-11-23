"""One text-in, text-out LLM call for every prompt in the pipeline.

Providers: groq, gemini (Google AI Studio), and two local OpenAI-compatible servers, ollama and
lmstudio (no key, no daily quota). Settings come from backend/.env; everything
except the keys is optional:

    GROQ_API_KEY=...         GEMINI_API_KEY=...
    LLM_PROVIDER=groq        # groq (default when GROQ_API_KEY is set), gemini, ollama or lmstudio
    LLM_MODEL=...            # overrides the provider's default model
    LLM_MAX_TOKENS=900       # cap on the answer length; some models allow very little per minute
    GROQ_API_URL=...         # any OpenAI-compatible chat/completions endpoint
    GEMINI_RPM=10            # requests per minute to stay under (free tier: 15 for Flash Lite)

Translation (pipeline.py) and shortening (fit_blocks.py, speech.py) can use
different LLMs; these override LLM_PROVIDER / LLM_MODEL for that job only:

    TRANSLATE_LLM_PROVIDER=...   TRANSLATE_LLM_MODEL=...
    FIT_LLM_PROVIDER=...         FIT_LLM_MODEL=...   (also used by speech.py)
    REVIEW_LLM_PROVIDER=...      REVIEW_LLM_MODEL=...   (optional second pass over the translation, see pipeline.py;
                                                         off while REVIEW_LLM_PROVIDER is empty)
    TRANSLATE_LLM_REASONING=...  FIT_LLM_REASONING=...  REVIEW_LLM_REASONING=...
        (e.g. "low" or "none": how hard a Groq / Gemini model may think; sent as reasoning_effort, empty = default)

Translation and shortening can also fall back to another LLM when their main one
runs out of its daily limit (for the rest of that run; later runs try the main LLM first again):

    TRANSLATE_FALLBACK_PROVIDER=gemini   TRANSLATE_FALLBACK_MODEL=gemini-3.5-flash-lite
    FIT_FALLBACK_PROVIDER=ollama         FIT_FALLBACK_MODEL=qwen3:8b

Local servers:

    LOCAL_LLM_URL=...        # default http://localhost:11434/v1 (ollama) or :1234/v1 (lmstudio)
    LOCAL_LLM_TIMEOUT=600    # seconds per answer; a local model on a busy GPU can be slow
    LOCAL_LLM_THINK=0        # 1 lets Qwen3 "think" first (slower; off by default)

Run `python llm.py` to check the local logic and make one real call,
`python llm.py --list [groq|gemini]` to list the models your key can use, or
`python llm.py --limits [MODEL]` to see a Groq model's real limits (one tiny request).
"""

import os
import random
import re
import sys
import threading
import time

import httpx
from dotenv import load_dotenv

from workdir import SCRIPT_DIR


load_dotenv(SCRIPT_DIR / ".env")

GROQ_URL = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")
GEMINI_BASE = os.getenv("GEMINI_API_URL", "https://generativelanguage.googleapis.com/v1beta/openai")
GEMINI_DEFAULT_RPM = 10
DEFAULT_MODELS = {
    "groq": "openai/gpt-oss-120b",
    "gemini": "gemini-3.5-flash-lite",
    "ollama": "qwen3:8b",
    "lmstudio": "qwen3-8b",
}
KEY_NAMES = {"groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY"}
# OpenAI-compatible servers on this computer: no key needed, no quota.
LOCAL_URLS = {
    "ollama": "http://localhost:11434/v1",
    "lmstudio": "http://localhost:1234/v1",
}
LOCAL_TIMEOUT = 600
LOCAL_ATTEMPTS = 3
# Settings that one job can override: translate = pipeline.py, fit = shortening,
# review = the optional second pass that checks the translation (off unless REVIEW_LLM_PROVIDER is set).
ROLE_PREFIXES = {"translate": "TRANSLATE_", "fit": "FIT_", "review": "REVIEW_"}
TEMPERATURE = 0.3
MAX_ATTEMPTS = 6
MAX_WAIT_SECONDS = 90  # a longer "try again in ..." is treated as an exhausted quota
REQUEST_TIMEOUT = 120


class QuotaExhausted(Exception):
    """A daily (or very long) limit was hit; waiting a few seconds will not help."""


class LLMError(RuntimeError):
    """The provider refused or failed the request for a reason retrying will not fix."""


# ---------------------------------------------------------
# Pure helpers (covered by run_local_checks)
# ---------------------------------------------------------

def strip_thinking(text):
    """Drop the <think>...</think> part that reasoning models put before the answer.

    Also handles a lone closing tag (the opening one was in the prompt template)
    and an answer cut off while still thinking (nothing usable: returns "").
    """
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    if "<think>" in text:
        text = text.split("<think>", 1)[0]
    return text.strip()


def is_daily_limit(message):
    lowered = message.lower()
    return any(marker in lowered for marker in ("per day", "(rpd)", "(tpd)", "perday"))


def parse_duration(message):
    """Seconds from text like 'try again in 7.2s', '12m3.5s' or '1h2m3s'; None if absent."""
    match = re.search(r"try again in\s*(?:(\d+)h)?\s*(?:(\d+)m(?![a-z]))?\s*(?:([\d.]+)s)?", message)
    if not match or not any(match.groups()):
        return None
    hours, minutes, seconds = match.groups()
    return int(hours or 0) * 3600 + int(minutes or 0) * 60 + float(seconds or 0)


def retry_seconds(headers, message):
    """How long the provider asked us to wait: Retry-After header first, then the message."""
    value = headers.get("retry-after")
    if value is not None:
        try:
            return float(value)
        except ValueError:
            pass
    return parse_duration(message)


def backoff(attempt):
    return min(60.0, 2 ** attempt + random.uniform(0, 1))


def is_local(provider):
    return provider in LOCAL_URLS


def settings(provider=None, model=None, role=None):
    """Return (provider, model, api_key) from the arguments, then .env, then defaults.

    `role` ("translate", "fit" or "review") first looks at TRANSLATE_LLM_* / FIT_LLM_* / REVIEW_LLM_*.
    A provider chosen without a model gets that provider's default model.
    Local providers need no key (api_key is LOCAL_LLM_API_KEY or "local").
    """
    prefix = ROLE_PREFIXES.get(role, "")
    if role is not None and role not in ROLE_PREFIXES:
        raise ValueError(f"Unknown LLM role {role!r}")
    explicit = provider is not None
    role_provider = os.getenv(f"{prefix}LLM_PROVIDER") if prefix else None
    if not explicit and role_provider:
        provider, explicit = role_provider, True
        if model is None:
            model = os.getenv(f"{prefix}LLM_MODEL")
    if model is None and not explicit and prefix:
        model = os.getenv(f"{prefix}LLM_MODEL")
    provider = (provider or os.getenv("LLM_PROVIDER")
                or ("groq" if os.getenv("GROQ_API_KEY") else "gemini")).strip().lower()
    if provider not in DEFAULT_MODELS:
        raise ValueError(f"LLM_PROVIDER must be one of {', '.join(DEFAULT_MODELS)}, got {provider!r}")
    if model is None and not explicit:
        model = os.getenv("LLM_MODEL")
    model = model or DEFAULT_MODELS[provider]
    if is_local(provider):
        return provider, model, os.getenv("LOCAL_LLM_API_KEY") or "local"
    key = os.getenv(KEY_NAMES[provider])
    if not key:
        raise RuntimeError(f"{KEY_NAMES[provider]} is missing from .env")
    return provider, model, key


def role_configured(role):
    """True when .env names an LLM for `role` of its own (REVIEW_LLM_PROVIDER=...)."""
    prefix = ROLE_PREFIXES.get(role)
    return bool(prefix and (os.getenv(f"{prefix}LLM_PROVIDER") or "").strip())


def reasoning_setting(role):
    """How hard the role's model may think ("low", "none"...), from {ROLE}_LLM_REASONING; None when unset."""
    prefix = ROLE_PREFIXES.get(role)
    value = (os.getenv(f"{prefix}LLM_REASONING") or "").strip().lower() if prefix else ""
    return value or None


FALLBACK_ROLES = ("translate", "fit")


def fallback_settings(role):
    """(provider, model, api_key) of the role's fallback LLM, or None when none is set.

    Translation and shortening can have one: TRANSLATE_FALLBACK_PROVIDER / FIT_FALLBACK_PROVIDER
    (+ ..._FALLBACK_MODEL; empty = that provider's default model).
    """
    if role not in FALLBACK_ROLES:
        return None
    prefix = ROLE_PREFIXES[role]
    provider = (os.getenv(f"{prefix}FALLBACK_PROVIDER") or "").strip().lower()
    if not provider:
        return None
    return settings(provider, os.getenv(f"{prefix}FALLBACK_MODEL") or None)


def describe(provider=None, model=None, role=None):
    """'groq:model-name' -- goes into cache keys so a model change re-translates."""
    chosen_provider, chosen_model, _ = settings(provider, model, role)
    return f"{chosen_provider}:{chosen_model}"


def local_url(provider):
    return (os.getenv("LOCAL_LLM_URL") or LOCAL_URLS[provider]).rstrip("/")


def local_timeout():
    value = os.getenv("LOCAL_LLM_TIMEOUT")
    try:
        return float(value) if value else LOCAL_TIMEOUT
    except ValueError:
        raise ValueError(f"LOCAL_LLM_TIMEOUT must be a number of seconds, got {value!r}") from None


def local_prompt(prompt, model):
    """Qwen3 thinks before answering unless told not to; that is slow on a laptop."""
    thinking = os.getenv("LOCAL_LLM_THINK", "0").strip().lower() in ("1", "true", "yes", "on")
    if "qwen3" in model.lower() and not thinking:
        return prompt.rstrip() + "\n\n/no_think"
    return prompt


# ---------------------------------------------------------
# Providers
# ---------------------------------------------------------

def _error_message(response):
    try:
        return str(response.json()["error"]["message"])
    except (ValueError, KeyError, TypeError):
        return response.text[:300]


def max_tokens_setting():
    value = os.getenv("LLM_MAX_TOKENS")
    if not value:
        return None
    if not value.isdigit() or int(value) <= 0:
        raise ValueError(f"LLM_MAX_TOKENS must be a positive whole number, got {value!r}")
    return int(value)


def _groq(prompt, model, api_key, post, sleep, reasoning=None):
    headers = {"Authorization": f"Bearer {api_key}"}
    body = {
        "model": model,
        "temperature": TEMPERATURE,
        "messages": [{"role": "user", "content": prompt}],
    }
    if max_tokens_setting():
        body["max_completion_tokens"] = max_tokens_setting()
    if reasoning:
        body["reasoning_effort"] = reasoning
    for attempt in range(1, MAX_ATTEMPTS + 1):
        last = attempt == MAX_ATTEMPTS
        try:
            response = post(GROQ_URL, headers=headers, json=body, timeout=REQUEST_TIMEOUT)
        except httpx.TransportError as error:  # timeouts, dropped connections
            if last:
                raise LLMError(f"Groq is unreachable: {error}") from error
            delay = backoff(attempt)
            print(f"Groq connection problem ({error.__class__.__name__}). "
                  f"Retry {attempt + 1}/{MAX_ATTEMPTS} in {delay:.1f}s...", flush=True)
            sleep(delay)
            continue

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"] or ""

        message = _error_message(response)
        if response.status_code == 429 and "request too large" in message.lower():
            # Waiting never helps: the single request is bigger than the model's limit.
            raise LLMError(f"{message} Set LLM_MAX_TOKENS in .env below the limit it names.")
        if response.status_code == 429:
            wait = retry_seconds(response.headers, message)
            if is_daily_limit(message) or last or (wait is not None and wait > MAX_WAIT_SECONDS):
                raise QuotaExhausted(message)
            delay = (wait + 0.5) if wait is not None else backoff(attempt)
            print(f"Groq rate limit (429). Retry {attempt + 1}/{MAX_ATTEMPTS} "
                  f"in {delay:.1f}s... ({message[:200]})", flush=True)
            sleep(delay)
        elif response.status_code >= 500 and not last:
            delay = backoff(attempt)
            print(f"Groq busy ({response.status_code}). Retry {attempt + 1}/{MAX_ATTEMPTS} "
                  f"in {delay:.1f}s...", flush=True)
            sleep(delay)
        else:
            hint = " Run `python llm.py --list` to see the models your key can use." \
                if response.status_code == 404 else ""
            raise LLMError(f"Groq error {response.status_code}: {message}{hint}")
    raise LLMError("Groq gave no answer")  # unreachable: the last attempt always returns/raises


def gemini_rpm():
    value = os.getenv("GEMINI_RPM")
    if not value:
        return GEMINI_DEFAULT_RPM
    try:
        rpm = float(value)
    except ValueError:
        rpm = 0
    if rpm <= 0:
        raise ValueError(f"GEMINI_RPM must be a positive number, got {value!r}")
    return rpm


_last_request = {"gemini": None}


def throttle(provider, rpm, sleep, now=time.monotonic):
    """Wait so that this process sends at most `rpm` requests per minute to `provider`."""
    interval = 60.0 / rpm
    last = _last_request.get(provider)
    if last is not None:
        wait = last + interval - now()
        if wait > 0:
            sleep(wait)
    _last_request[provider] = now()


def gemini_error_info(response):
    """(message, daily, wait_seconds) from a Gemini error response.

    Google reports which quota was hit (a quotaId such as ...PerDay... or
    ...PerMinute...) and, for a per-minute limit, a RetryInfo retryDelay like "34s".
    """
    try:
        body = response.json()
    except ValueError:
        body = None
    if isinstance(body, list) and body:
        body = body[0]
    error = body.get("error", {}) if isinstance(body, dict) else {}
    message = str(error.get("message") or response.text[:300])
    details = error.get("details") if isinstance(error.get("details"), list) else []
    quota_ids = " ".join(
        str(violation.get("quotaId", "")) + " " + str(violation.get("quotaMetric", ""))
        for item in details if isinstance(item, dict)
        for violation in (item.get("violations") or []) if isinstance(violation, dict)
    )
    daily = "perday" in quota_ids.lower() or is_daily_limit(message)
    wait = None
    for item in details:
        delay = item.get("retryDelay") if isinstance(item, dict) else None
        if isinstance(delay, str) and re.fullmatch(r"[\d.]+s", delay):
            wait = float(delay[:-1])
    if wait is None:
        wait = parse_duration(message)
    return message, daily, wait


def _gemini(prompt, model, api_key, post, sleep, reasoning=None, max_tokens=None):
    """Google AI Studio through its OpenAI-compatible endpoint.

    A per-minute limit (RPM, TPM) is waited out; a per-day limit stops the run
    with QuotaExhausted, since waiting would not help.
    """
    url = GEMINI_BASE.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    body = {
        "model": model,
        "temperature": TEMPERATURE,
        "messages": [{"role": "user", "content": prompt}],
    }
    if max_tokens or max_tokens_setting():
        body["max_tokens"] = max_tokens or max_tokens_setting()
    if reasoning:
        body["reasoning_effort"] = reasoning
    rpm = gemini_rpm()
    for attempt in range(1, MAX_ATTEMPTS + 1):
        last = attempt == MAX_ATTEMPTS
        throttle("gemini", rpm, sleep)
        try:
            response = post(url, headers=headers, json=body, timeout=REQUEST_TIMEOUT)
        except httpx.TransportError as error:
            if last:
                raise LLMError(f"Gemini is unreachable: {error}") from error
            delay = backoff(attempt)
            print(f"Gemini connection problem ({error.__class__.__name__}). "
                  f"Retry {attempt + 1}/{MAX_ATTEMPTS} in {delay:.1f}s...", flush=True)
            sleep(delay)
            continue

        if response.status_code == 200:
            try:
                return response.json()["choices"][0]["message"]["content"] or ""
            except (ValueError, KeyError, IndexError, TypeError) as error:
                raise LLMError(f"Gemini sent an answer in an unexpected format: {response.text[:300]}") from error

        message, daily, wait = gemini_error_info(response)
        if response.status_code == 429:
            if daily:
                raise QuotaExhausted(
                    f"The daily Gemini quota for {model} is used up (the free tier resets at midnight "
                    f"Pacific time). Run the same command later, or set another model in .env "
                    f"(TRANSLATE_LLM_MODEL / FIT_LLM_MODEL). Google said: {message[:200]}")
            if last or (wait is not None and wait > MAX_WAIT_SECONDS):
                raise QuotaExhausted(f"Gemini rate limit for {model} did not clear: {message[:300]}")
            delay = (wait + 1.0) if wait is not None else max(backoff(attempt), 60.0 / rpm)
            print(f"Gemini rate limit (429). Retry {attempt + 1}/{MAX_ATTEMPTS} "
                  f"in {delay:.1f}s... ({message[:150]})", flush=True)
            sleep(delay)
        elif response.status_code >= 500 and not last:
            delay = backoff(attempt)
            print(f"Gemini busy ({response.status_code}). Retry {attempt + 1}/{MAX_ATTEMPTS} "
                  f"in {delay:.1f}s...", flush=True)
            sleep(delay)
        else:
            hint = (" Run `python llm.py --list gemini` to see the model names your key can use."
                    if response.status_code in (400, 404) else "")
            raise LLMError(f"Gemini error {response.status_code}: {message[:300]}{hint}")
    raise LLMError("Gemini gave no answer")  # unreachable


def _local(prompt, provider, model, api_key, post, sleep):
    """Ollama / LM Studio through their OpenAI-compatible endpoint.

    No rate-limit handling: a local server has no quota. Only a server that is
    not running (yet) or briefly busy is retried.
    """
    name = "Ollama" if provider == "ollama" else "LM Studio"
    url = local_url(provider) + "/chat/completions"
    body = {
        "model": model,
        "temperature": TEMPERATURE,
        "messages": [{"role": "user", "content": local_prompt(prompt, model)}],
    }
    if max_tokens_setting():
        body["max_tokens"] = max_tokens_setting()
    headers = {"Authorization": f"Bearer {api_key}"}
    for attempt in range(1, LOCAL_ATTEMPTS + 1):
        last = attempt == LOCAL_ATTEMPTS
        try:
            response = post(url, headers=headers, json=body, timeout=local_timeout())
        except httpx.TimeoutException as error:
            if last:
                raise LLMError(f"{name} did not answer within {local_timeout():.0f}s. A smaller model "
                               "or a larger LOCAL_LLM_TIMEOUT in .env may help.") from error
            print(f"{name} is slow ({error.__class__.__name__}). Retry {attempt + 1}/{LOCAL_ATTEMPTS}...",
                  flush=True)
            continue
        except httpx.TransportError as error:  # nothing listening, dropped connection
            if last:
                start = "Start the Ollama app (or run `ollama serve`)" if provider == "ollama" \
                    else "Start the LM Studio server (Developer tab)"
                raise LLMError(f"Cannot reach {name} at {local_url(provider)}: {error}. {start}, "
                               "or set LOCAL_LLM_URL in .env.") from error
            delay = backoff(attempt)
            print(f"{name} not reachable. Retry {attempt + 1}/{LOCAL_ATTEMPTS} in {delay:.1f}s...",
                  flush=True)
            sleep(delay)
            continue

        if response.status_code == 200:
            try:
                return response.json()["choices"][0]["message"]["content"] or ""
            except (ValueError, KeyError, IndexError, TypeError) as error:
                raise LLMError(f"{name} sent an answer in an unexpected format: {response.text[:300]}") from error

        message = _error_message(response)
        if response.status_code == 404 or "not found" in message.lower():
            hint = f" Run `ollama pull {model}` first." if provider == "ollama" \
                else " Load the model in LM Studio and copy its identifier into the model setting."
            raise LLMError(f"{name} has no model {model!r}: {message}.{hint}")
        if (response.status_code >= 500 or response.status_code == 429) and not last:
            delay = backoff(attempt)
            print(f"{name} error {response.status_code} ({message[:200]}). "
                  f"Retry {attempt + 1}/{LOCAL_ATTEMPTS} in {delay:.1f}s...", flush=True)
            sleep(delay)
            continue
        raise LLMError(f"{name} error {response.status_code}: {message}")
    raise LLMError(f"{name} gave no answer")  # unreachable


def check_local_server(role=None, get=httpx.get, fallback=False):
    """None when `role`'s LLM is not local or the server has the model; else a problem to show.

    Used before a long run so a stopped Ollama or a missing `ollama pull` shows up now,
    not after Whisper has finished. `fallback=True` checks the role's fallback LLM instead.
    """
    if fallback:
        chosen = fallback_settings(role)
        if chosen is None:
            return None
        provider, model, _ = chosen
    else:
        provider, model, _ = settings(role=role)
    if not is_local(provider):
        return None
    name = "Ollama" if provider == "ollama" else "LM Studio"
    try:
        response = get(local_url(provider) + "/models", timeout=10)
        names = [item.get("id", "") for item in response.json().get("data", [])]
    except (httpx.HTTPError, ValueError, AttributeError) as error:
        return (f"Cannot reach {name} at {local_url(provider)} ({error.__class__.__name__}). "
                + ("Start the Ollama app or run `ollama serve`." if provider == "ollama"
                   else "Start the LM Studio server (Developer tab)."))
    if provider == "ollama" and model not in names and f"{model}:latest" not in names:
        return f"Ollama has no model {model!r}. Run: ollama pull {model}"
    if provider == "lmstudio" and model not in names:
        return f"LM Studio has not loaded {model!r} (loaded: {', '.join(names) or 'none'})."
    return None


def list_models(provider="groq", get=httpx.get):
    """Model ids the provider's key can use, sorted (`python llm.py --list [gemini]`)."""
    _, _, api_key = settings(provider)
    base = GROQ_URL.rsplit("/chat/completions", 1)[0] if provider == "groq" else GEMINI_BASE.rstrip("/")
    response = get(base + "/models", headers={"Authorization": f"Bearer {api_key}"}, timeout=REQUEST_TIMEOUT)
    if response.status_code != 200:
        raise LLMError(f"{provider.capitalize()} error {response.status_code}: {_error_message(response)}")
    return sorted(item["id"].removeprefix("models/") for item in response.json()["data"])


list_groq_models = list_models  # the old name


_switched = set()  # roles that moved to their fallback LLM in this process
_switch_lock = threading.Lock()


def generate(prompt, model=None, provider=None, role=None, post=httpx.post, sleep=time.sleep, max_tokens=None):
    """Send one prompt and return the answer text (reasoning blocks removed).

    `role` ("translate", "fit" or "review") picks that job's settings, see settings().
    `max_tokens` overrides LLM_MAX_TOKENS for this one request (Gemini only; a whole-video answer is long).
    {ROLE}_LLM_REASONING (e.g. REVIEW_LLM_REASONING=low) is sent to Groq and Gemini as reasoning_effort.
    When the role's main LLM is out of quota and it has a fallback (TRANSLATE_FALLBACK_*, FIT_FALLBACK_*),
    this and every later call of the process use the fallback.
    """
    explicit = provider is not None or model is not None
    fallback = None if explicit else fallback_settings(role)
    reasoning = None if explicit else reasoning_setting(role)
    if fallback and role in _switched:
        return _generate_with(prompt, fallback, post, sleep, reasoning, max_tokens)
    try:
        return _generate_with(prompt, settings(provider, model, role), post, sleep, reasoning, max_tokens)
    except QuotaExhausted as error:
        if not fallback:
            raise
        with _switch_lock:
            if role not in _switched:
                _switched.add(role)
                print(f"{describe(role=role)} is out of quota ({str(error)[:120]}). "
                      f"Switching to {fallback[0]}:{fallback[1]} for the rest of this run.", flush=True)
        return _generate_with(prompt, fallback, post, sleep, reasoning, max_tokens)


def active_name(role):
    """'provider:model' that answers `role` now: its fallback once the main LLM ran out in this process."""
    fallback = fallback_settings(role) if role in _switched else None
    return f"{fallback[0]}:{fallback[1]}" if fallback else describe(role=role)


def _generate_with(prompt, chosen, post, sleep, reasoning=None, max_tokens=None):
    chosen_provider, chosen_model, api_key = chosen
    if chosen_provider == "groq":
        text = _groq(prompt, chosen_model, api_key, post, sleep, reasoning)
    elif is_local(chosen_provider):
        text = _local(prompt, chosen_provider, chosen_model, api_key, post, sleep)
    else:
        text = _gemini(prompt, chosen_model, api_key, post, sleep, reasoning, max_tokens)
    return strip_thinking(text)


RATE_HEADERS = (
    ("x-ratelimit-limit-requests", "requests per day"),
    ("x-ratelimit-remaining-requests", "requests left today"),
    ("x-ratelimit-reset-requests", "requests reset in"),
    ("x-ratelimit-limit-tokens", "tokens per minute"),
    ("x-ratelimit-remaining-tokens", "tokens left this minute"),
    ("x-ratelimit-reset-tokens", "tokens reset in"),
)


def describe_limits(headers):
    """Readable lines from Groq's x-ratelimit-* response headers ([] when there are none)."""
    return [f"{label}: {headers[name]}" for name, label in RATE_HEADERS if headers.get(name) is not None]


def groq_limits(model=None, post=httpx.post):
    """One tiny request to `model` (default: the translation model) and the limits Groq reports with it."""
    if model is None:
        provider, model, api_key = settings(role="translate")
        if provider != "groq":
            raise LLMError(f"Translation uses {provider}, not Groq; name a Groq model: python llm.py --limits MODEL")
    else:
        _, _, api_key = settings("groq", model)
    body = {"model": model, "messages": [{"role": "user", "content": "Reply with exactly one word: ok"}],
            "max_completion_tokens": 64}
    response = post(GROQ_URL, headers={"Authorization": f"Bearer {api_key}"}, json=body, timeout=REQUEST_TIMEOUT)
    lines = describe_limits(response.headers)
    if response.status_code != 200 and not lines:
        raise LLMError(f"Groq error {response.status_code}: {_error_message(response)}")
    return model, response.status_code, lines


# ---------------------------------------------------------
# Checks
# ---------------------------------------------------------

class _FakeResponse:
    def __init__(self, status, content="", error=None, headers=None):
        self.status_code = status
        self.headers = headers or {}
        self._body = ({"choices": [{"message": {"content": content}}]} if status == 200
                      else {"error": {"message": error or "failed"}})
        self.text = str(self._body)

    def json(self):
        return self._body


def _scripted_post(*responses):
    queue = list(responses)
    post_calls = []

    def post(url, **kwargs):
        post.url = url
        post_calls.append(kwargs)
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item
    post.calls = post_calls
    return post


def _check_fallback(wait):
    """Quota on the main LLM moves the "fit" role to FIT_FALLBACK_*, once, and keeps it there."""
    day = _FakeResponse(429, error="Rate limit ... on tokens per day (TPD): Limit 200000. Please try again in 6m")
    os.environ["FIT_LLM_PROVIDER"], os.environ["FIT_LLM_MODEL"] = "groq", "big"
    # No fallback set: the quota error still reaches the caller.
    assert fallback_settings("fit") is None and fallback_settings("translate") is None
    try:
        generate("hi", role="fit", post=_scripted_post(day), sleep=wait)
    except QuotaExhausted:
        pass
    else:
        raise AssertionError("without a fallback the quota error must surface")
    assert not _switched

    os.environ["FIT_FALLBACK_PROVIDER"], os.environ["FIT_FALLBACK_MODEL"] = "ollama", "qwen3:8b"
    assert fallback_settings("fit")[:2] == ("ollama", "qwen3:8b") and fallback_settings("translate") is None
    post = _scripted_post(day, _FakeResponse(200, "from fallback"))
    assert generate("hi", role="fit", post=post, sleep=wait) == "from fallback"
    assert "11434" in post.url and post.calls[1]["json"]["model"] == "qwen3:8b" and "fit" in _switched
    post = _scripted_post(_FakeResponse(200, "again"))
    assert generate("hi", role="fit", post=post, sleep=wait) == "again" and len(post.calls) == 1
    assert "11434" in post.url  # later calls go straight to the fallback
    # Other roles and explicit arguments are not affected.
    assert generate("hi", role="translate", post=_scripted_post(_FakeResponse(200, "t")), sleep=wait) == "t"
    _switched.clear()
    os.environ.pop("FIT_FALLBACK_PROVIDER"); os.environ.pop("FIT_FALLBACK_MODEL")
    os.environ.pop("FIT_LLM_PROVIDER"); os.environ.pop("FIT_LLM_MODEL")

    # Translation has its own fallback (Qwen on Groq out of tokens -> Gemini); review has none.
    os.environ["TRANSLATE_LLM_PROVIDER"], os.environ["TRANSLATE_LLM_MODEL"] = "groq", "qwen/qwen3.8-27b"
    os.environ["TRANSLATE_FALLBACK_PROVIDER"] = "ollama"  # no model: the provider's default
    assert fallback_settings("translate")[:2] == ("ollama", DEFAULT_MODELS["ollama"])
    assert fallback_settings("fit") is None and fallback_settings("review") is None
    post = _scripted_post(day, _FakeResponse(200, "dịch dự phòng"))
    assert generate("hi", role="translate", post=post, sleep=wait) == "dịch dự phòng"
    assert post.calls[0]["json"]["model"] == "qwen/qwen3.8-27b" and "translate" in _switched
    assert active_name("translate") == f"ollama:{DEFAULT_MODELS['ollama']}"
    _switched.clear()
    assert active_name("translate") == "groq:qwen/qwen3.8-27b"
    os.environ.pop("TRANSLATE_FALLBACK_PROVIDER")
    os.environ.pop("TRANSLATE_LLM_PROVIDER"); os.environ.pop("TRANSLATE_LLM_MODEL")

    headers = {"x-ratelimit-limit-requests": "1000", "x-ratelimit-limit-tokens": "8000",
               "x-ratelimit-remaining-tokens": "7990"}
    assert describe_limits(headers) == ["requests per day: 1000", "tokens per minute: 8000",
                                        "tokens left this minute: 7990"]
    assert describe_limits({}) == []
    post = _scripted_post(_FakeResponse(200, "ok", headers=headers))
    assert groq_limits("qwen/qwen3.8-27b", post=post)[1:] == (200, describe_limits(headers))
    assert post.calls[0]["json"]["model"] == "qwen/qwen3.8-27b"


def _check_roles_and_local():
    """Role overrides and the local provider (called with a clean environment)."""
    sleeps = []
    groq_default = ("groq", DEFAULT_MODELS["groq"])
    assert settings(role="translate")[:2] == groq_default
    os.environ["LLM_MODEL"] = "groq-model"
    os.environ["FIT_LLM_PROVIDER"] = "ollama"
    assert settings(role="fit")[:2] == ("ollama", "qwen3:8b")  # LLM_MODEL belongs to groq
    assert settings(role="translate")[:2] == ("groq", "groq-model")
    assert settings()[:2] == ("groq", "groq-model")
    os.environ["FIT_LLM_MODEL"] = "qwen2.5:7b-instruct"
    assert describe(role="fit") == "ollama:qwen2.5:7b-instruct"
    os.environ["TRANSLATE_LLM_MODEL"] = "other-groq-model"  # model only: provider stays LLM_PROVIDER
    assert settings(role="translate")[:2] == ("groq", "other-groq-model")
    assert settings("groq", "x", role="fit")[:2] == ("groq", "x")  # arguments win
    try:
        settings(role="tts")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown role must be rejected")
    for name in ("LLM_MODEL", "FIT_LLM_PROVIDER", "FIT_LLM_MODEL", "TRANSLATE_LLM_MODEL"):
        os.environ.pop(name)

    # The review role exists but is off until REVIEW_LLM_PROVIDER is set; reasoning is opt-in.
    assert not role_configured("review") and not role_configured("tts") and reasoning_setting("review") is None
    os.environ["REVIEW_LLM_PROVIDER"], os.environ["REVIEW_LLM_MODEL"] = "groq", "review-model"
    os.environ["REVIEW_LLM_REASONING"] = " Low "
    assert role_configured("review") and describe(role="review") == "groq:review-model"
    assert reasoning_setting("review") == "low" and reasoning_setting("fit") is None
    post = _scripted_post(_FakeResponse(200, "ok"))
    assert generate("hi", role="review", post=post, sleep=sleeps.append) == "ok"
    assert post.calls[0]["json"]["reasoning_effort"] == "low" and post.calls[0]["json"]["model"] == "review-model"
    post = _scripted_post(_FakeResponse(200, "ok"))
    generate("hi", role="translate", post=post, sleep=sleeps.append)
    assert "reasoning_effort" not in post.calls[0]["json"]
    for name in ("REVIEW_LLM_PROVIDER", "REVIEW_LLM_MODEL", "REVIEW_LLM_REASONING"):
        os.environ.pop(name)

    # Local: no key, Qwen3 told not to think, right URL, no quota logic.
    os.environ.pop("GROQ_API_KEY")
    assert settings("ollama")[2] == "local"
    post = _scripted_post(_FakeResponse(200, "<think></think>Chào"))
    assert generate("hi", provider="ollama", post=post, sleep=sleeps.append) == "Chào"
    call = post.calls[0]
    assert call["json"]["model"] == "qwen3:8b"
    assert call["json"]["messages"][0]["content"].endswith("/no_think")
    assert post.url == "http://localhost:11434/v1/chat/completions"
    os.environ["LOCAL_LLM_THINK"] = "1"
    post = _scripted_post(_FakeResponse(200, "ok"))
    generate("hi", provider="ollama", post=post, sleep=sleeps.append)
    assert "/no_think" not in post.calls[0]["json"]["messages"][0]["content"]
    os.environ.pop("LOCAL_LLM_THINK")
    post = _scripted_post(_FakeResponse(200, "ok"))
    generate("hi", provider="lmstudio", model="m", post=post, sleep=sleeps.append)
    assert post.url == "http://localhost:1234/v1/chat/completions"
    assert "/no_think" not in post.calls[0]["json"]["messages"][0]["content"]
    os.environ["LOCAL_LLM_URL"] = "http://127.0.0.1:9999/v1/"
    post = _scripted_post(_FakeResponse(200, "ok"))
    generate("hi", provider="ollama", post=post, sleep=sleeps.append)
    assert post.url == "http://127.0.0.1:9999/v1/chat/completions"
    os.environ.pop("LOCAL_LLM_URL")

    # A local 429 / 500 is just "busy": retried, never QuotaExhausted.
    sleeps.clear()
    post = _scripted_post(_FakeResponse(429, error="tokens per day"), _FakeResponse(500, error="x"),
                          _FakeResponse(200, "ok"))
    assert generate("hi", provider="ollama", post=post, sleep=sleeps.append) == "ok"
    assert len(sleeps) == 2
    for failure, words in ((_FakeResponse(404, error='model "qwen3:8b" not found'), "ollama pull"),):
        try:
            generate("hi", provider="ollama", post=_scripted_post(failure), sleep=sleeps.append)
        except LLMError as error:
            assert words in str(error), error
        else:
            raise AssertionError("a missing model must stop with a hint")
    refused = httpx.ConnectError("refused")
    try:
        generate("hi", provider="ollama", post=_scripted_post(refused, refused, refused), sleep=sleeps.append)
    except LLMError as error:
        assert "ollama serve" in str(error)
    else:
        raise AssertionError("an unreachable server must stop with a hint")
    os.environ["GROQ_API_KEY"] = "test-key"


class _GeminiResponse:
    def __init__(self, status, content="ok", error=None, details=None, as_list=False):
        self.status_code = status
        self.headers = {}
        if status == 200:
            self._body = {"choices": [{"message": {"content": content}}]}
        else:
            body = {"error": {"code": status, "message": error or "failed", "details": details or []}}
            self._body = [body] if as_list else body
        self.text = str(self._body)

    def json(self):
        return self._body


def _quota(quota_id, retry="34s"):
    return [
        {"@type": "type.googleapis.com/google.rpc.QuotaFailure",
         "violations": [{"quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
                         "quotaId": quota_id}]},
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry},
    ]


def _check_gemini():
    """Gemini: per-minute limits are waited out, per-day limits stop the run."""
    sleeps = []
    os.environ["GEMINI_API_KEY"] = "gemini-key"
    os.environ["GEMINI_RPM"] = "600000"  # no real waiting in these checks
    assert settings("gemini")[:2] == ("gemini", DEFAULT_MODELS["gemini"])
    assert "flash-lite" in DEFAULT_MODELS["gemini"]

    post = _scripted_post(_GeminiResponse(200, "<think>x</think>Chào"))
    assert generate("hi", provider="gemini", post=post, sleep=sleeps.append) == "Chào"
    assert post.url == GEMINI_BASE + "/chat/completions"
    assert post.calls[0]["headers"] == {"Authorization": "Bearer gemini-key"}

    per_minute = _GeminiResponse(429, error="Quota exceeded for metric ... PerMinute",
                                 details=_quota("GenerateRequestsPerMinutePerProjectPerModel-FreeTier"),
                                 as_list=True)
    post = _scripted_post(per_minute, _GeminiResponse(200, "ok"))
    assert generate("hi", provider="gemini", post=post, sleep=sleeps.append) == "ok"
    assert 35.0 in sleeps, sleeps  # retryDelay 34s + 1s

    per_day = _GeminiResponse(429, error="Quota exceeded ...",
                              details=_quota("GenerateRequestsPerDayPerProjectPerModel-FreeTier", "20s"))
    post = _scripted_post(per_day)
    try:
        generate("hi", model="gemini-x", provider="gemini", post=post, sleep=sleeps.append)
    except QuotaExhausted as error:
        assert "daily" in str(error) and "gemini-x" in str(error) and len(post.calls) == 1
    else:
        raise AssertionError("a daily Gemini limit must stop without retrying")

    long_wait = _GeminiResponse(429, error="slow down", details=_quota("GenerateRequestsPerMinutePerProjectPerModel", "300s"))
    try:
        generate("hi", provider="gemini", post=_scripted_post(long_wait), sleep=sleeps.append)
    except QuotaExhausted:
        pass
    else:
        raise AssertionError("a very long wait counts as exhausted")

    # No details at all: the message text decides, then plain backoff.
    post = _scripted_post(_GeminiResponse(429, error="Rate limit, please try again in 7.5s"), _GeminiResponse(200, "ok"))
    sleeps.clear()
    assert generate("hi", provider="gemini", post=post, sleep=sleeps.append) == "ok"
    assert [wait for wait in sleeps if wait >= 1] == [8.5], sleeps

    sleeps.clear()
    post = _scripted_post(_GeminiResponse(503, error="busy"), httpx.ReadTimeout("slow"), _GeminiResponse(200, "ok"))
    assert generate("hi", provider="gemini", post=post, sleep=sleeps.append) == "ok"
    assert len([wait for wait in sleeps if wait >= 1]) == 2, sleeps
    try:
        generate("hi", provider="gemini", post=_scripted_post(_GeminiResponse(404, error="model not found")),
                 sleep=sleeps.append)
    except LLMError as error:
        assert "--list gemini" in str(error)
    else:
        raise AssertionError("an unknown model must stop with a hint")

    os.environ["LLM_MAX_TOKENS"] = "700"
    post = _scripted_post(_GeminiResponse(200))
    generate("hi", provider="gemini", post=post, sleep=sleeps.append)
    assert post.calls[0]["json"]["max_tokens"] == 700
    post = _scripted_post(_GeminiResponse(200))
    generate("hi", provider="gemini", post=post, sleep=sleeps.append, max_tokens=32000)
    assert post.calls[0]["json"]["max_tokens"] == 32000  # a per-request cap beats LLM_MAX_TOKENS
    os.environ.pop("LLM_MAX_TOKENS")

    # Spacing: at 10 requests a minute, a second request waits for the rest of 6 seconds.
    waits, clock = [], [100.0]
    _last_request["gemini"] = None
    throttle("gemini", 10, waits.append, now=lambda: clock[0])
    clock[0] = 102.0
    throttle("gemini", 10, waits.append, now=lambda: clock[0])
    assert waits == [4.0], waits
    clock[0] = 200.0
    throttle("gemini", 10, waits.append, now=lambda: clock[0])
    assert waits == [4.0]
    os.environ["GEMINI_RPM"] = "0"
    try:
        gemini_rpm()
    except ValueError:
        pass
    else:
        raise AssertionError("GEMINI_RPM must be positive")
    assert _gemini_models_listing_ok()
    os.environ.pop("GEMINI_RPM")
    os.environ.pop("GEMINI_API_KEY")
    _last_request["gemini"] = None


def _gemini_models_listing_ok():
    class Reply:
        status_code = 200
        def json(self):
            return {"data": [{"id": "models/gemini-b"}, {"id": "models/gemini-a"}]}
    return list_models("gemini", get=lambda url, **kw: Reply()) == ["gemini-a", "gemini-b"]


def run_local_checks():
    assert strip_thinking("<think>hmm\nmore</think>\nXin chào") == "Xin chào"
    assert strip_thinking(None) == ""
    assert strip_thinking("reasoning only\n</think>\n[1]") == "[1]"
    assert strip_thinking("<think>cut off mid-thought") == ""
    assert parse_duration("Please try again in 7.2s.") == 7.2
    assert parse_duration("Please try again in 12m3.5s.") == 723.5
    assert parse_duration("Please try again in 1h2m3s") == 3723
    assert parse_duration("Please try again in 2m") == 120
    assert parse_duration("no hint here") is None
    assert is_daily_limit("Rate limit reached ... on requests per day (RPD): Limit 1000")
    assert not is_daily_limit("Rate limit reached ... on tokens per minute (TPM)")
    assert retry_seconds({"retry-after": "3"}, "") == 3.0
    assert retry_seconds({}, "try again in 4s") == 4.0
    assert retry_seconds({"retry-after": "soon"}, "try again in 4s") == 4.0

    names = ("GROQ_API_KEY", "LLM_PROVIDER", "LLM_MODEL", "LLM_MAX_TOKENS",
             "TRANSLATE_LLM_PROVIDER", "TRANSLATE_LLM_MODEL", "FIT_LLM_PROVIDER", "FIT_LLM_MODEL",
             "FIT_FALLBACK_PROVIDER", "FIT_FALLBACK_MODEL", "TRANSLATE_FALLBACK_PROVIDER", "TRANSLATE_FALLBACK_MODEL",
             "LOCAL_LLM_URL", "LOCAL_LLM_TIMEOUT", "LOCAL_LLM_THINK", "LOCAL_LLM_API_KEY",
             "GEMINI_API_KEY", "GEMINI_RPM")
    saved = {name: os.environ.get(name) for name in names}
    for name in names:
        os.environ.pop(name, None)
    os.environ["GROQ_API_KEY"] = "test-key"
    try:
        _check_roles_and_local()
        os.environ["GROQ_API_KEY"] = "test-key"
        _check_gemini()
        assert settings()[:2] == ("groq", DEFAULT_MODELS["groq"])
        assert settings("groq", "other-model")[1] == "other-model"
        assert describe() == f"groq:{DEFAULT_MODELS['groq']}"
        sleeps = []
        wait = sleeps.append

        reply = generate("hi", post=_scripted_post(_FakeResponse(200, "<think>x</think>Chào")), sleep=wait)
        assert reply == "Chào" and not sleeps

        limited = _FakeResponse(429, error="Rate limit ... tokens per minute (TPM)",
                                headers={"retry-after": "2"})
        reply = generate("hi", post=_scripted_post(limited, _FakeResponse(200, "ok")), sleep=wait)
        assert reply == "ok" and sleeps == [2.5]

        for error in ("Rate limit ... requests per day (RPD). Please try again in 3m",
                      "Rate limit ... tokens per minute. Please try again in 5m"):
            try:
                generate("hi", post=_scripted_post(_FakeResponse(429, error=error)), sleep=wait)
            except QuotaExhausted:
                pass
            else:
                raise AssertionError(f"429 should stop: {error}")

        sleeps.clear()
        reply = generate("hi", post=_scripted_post(_FakeResponse(503, error="busy"),
                                                    httpx.ReadTimeout("slow"),
                                                    _FakeResponse(200, "ok")), sleep=wait)
        assert reply == "ok" and len(sleeps) == 2

        too_large = _FakeResponse(429, error="Request too large for model x on output tokens per minute "
                                  "(OTPM): Limit 1000, Requested 2048.")
        post = _scripted_post(too_large)
        try:
            generate("hi", post=post, sleep=wait)
        except LLMError as error:
            assert "LLM_MAX_TOKENS" in str(error) and len(post.calls) == 1  # not retried
        else:
            raise AssertionError("an oversized request must not be retried")

        os.environ["LLM_MAX_TOKENS"] = "900"
        post = _scripted_post(_FakeResponse(200, "ok"))
        generate("hi", post=post, sleep=wait)
        assert post.calls[0]["json"]["max_completion_tokens"] == 900
        os.environ["LLM_MAX_TOKENS"] = "lots"
        try:
            generate("hi", post=_scripted_post(_FakeResponse(200, "ok")), sleep=wait)
        except ValueError:
            pass
        else:
            raise AssertionError("a non-numeric LLM_MAX_TOKENS must be rejected")
        os.environ.pop("LLM_MAX_TOKENS")

        _check_fallback(wait)

        try:
            generate("hi", post=_scripted_post(_FakeResponse(400, error="bad model")), sleep=wait)
        except LLMError as error:
            assert "bad model" in str(error)
        else:
            raise AssertionError("a 400 must not be retried")
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


if __name__ == "__main__":
    run_local_checks()
    print("llm local checks passed")
    if "--limits" in sys.argv[1:]:
        rest = [arg for arg in sys.argv[1:] if arg != "--limits"]
        try:
            name, status, lines = groq_limits(rest[0] if rest else None)
        except (LLMError, RuntimeError, ValueError) as error:
            raise SystemExit(f"Failed: {error}")
        print(f"Groq limits for {name} (answer status {status}):")
        print("\n".join("  " + line for line in lines) or "  Groq sent no x-ratelimit headers.")
        raise SystemExit
    if "--list" in sys.argv[1:]:
        names = [arg for arg in sys.argv[1:] if arg in ("groq", "gemini")]
        print("\n".join(list_models(names[0] if names else "groq")))
        raise SystemExit
    for role in ("translate", "fit"):
        try:
            provider, model, _ = settings(role=role)
        except (RuntimeError, ValueError) as error:
            raise SystemExit(f"Skipping the live call: {error}")
        print(f"Live call for {role} with {provider}:{model} ...")
        try:
            print(generate("Reply with exactly one word: ok", role=role))
        except (LLMError, QuotaExhausted) as error:
            raise SystemExit(f"Failed: {error}")
