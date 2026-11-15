"""One text-in, text-out LLM call for every prompt in the pipeline (Groq or Gemini).

Settings come from backend/.env; everything except the keys is optional:

    GROQ_API_KEY=...         GEMINI_API_KEY=...
    LLM_PROVIDER=groq        # groq (default when GROQ_API_KEY is set) or gemini
    LLM_MODEL=...            # overrides the provider's default model
    LLM_MAX_TOKENS=900       # cap on the answer length; some models allow very little per minute
    GROQ_API_URL=...         # any OpenAI-compatible chat/completions endpoint

Run `python llm.py` to check the local logic and make one real call,
or `python llm.py --list` to list the models your Groq key can use.
"""

import os
import random
import re
import sys
import time

import httpx
from dotenv import load_dotenv

from workdir import SCRIPT_DIR


load_dotenv(SCRIPT_DIR / ".env")

GROQ_URL = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")
DEFAULT_MODELS = {
    "groq": "openai/gpt-oss-120b",
    "gemini": "gemini-3.6-flash",
}
KEY_NAMES = {"groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY"}
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
    """Drop <think>...</think> blocks that reasoning models put before the answer."""
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


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


def settings(provider=None, model=None):
    """Return (provider, model, api_key) from the arguments, then .env, then defaults."""
    explicit = provider is not None
    provider = (provider or os.getenv("LLM_PROVIDER")
                or ("groq" if os.getenv("GROQ_API_KEY") else "gemini")).lower()
    if provider not in DEFAULT_MODELS:
        raise ValueError(f"LLM_PROVIDER must be one of {', '.join(DEFAULT_MODELS)}, got {provider!r}")
    if model is None and not explicit:
        model = os.getenv("LLM_MODEL")
    model = model or DEFAULT_MODELS[provider]
    key = os.getenv(KEY_NAMES[provider])
    if not key:
        raise RuntimeError(f"{KEY_NAMES[provider]} is missing from .env")
    return provider, model, key


def describe(provider=None, model=None):
    """'groq:model-name' -- goes into cache keys so a model change re-translates."""
    chosen_provider, chosen_model, _ = settings(provider, model)
    return f"{chosen_provider}:{chosen_model}"


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


def _groq(prompt, model, api_key, post, sleep):
    headers = {"Authorization": f"Bearer {api_key}"}
    body = {
        "model": model,
        "temperature": TEMPERATURE,
        "messages": [{"role": "user", "content": prompt}],
    }
    if max_tokens_setting():
        body["max_completion_tokens"] = max_tokens_setting()
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


def _gemini(prompt, model, api_key, sleep):
    from google import genai
    from google.genai import errors

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with genai.Client(api_key=api_key) as client:
                return client.models.generate_content(model=model, contents=prompt).text or ""
        except errors.ServerError as error:
            if error.code != 503 or attempt == MAX_ATTEMPTS:
                raise LLMError(f"Gemini error {error.code}: {error}") from error
            delay = backoff(attempt)
            print(f"Gemini busy (503). Retry {attempt + 1}/{MAX_ATTEMPTS} "
                  f"in {delay:.1f}s...", flush=True)
            sleep(delay)
        except errors.ClientError as error:
            if error.code == 429:
                raise QuotaExhausted(str(error)[:300]) from error
            raise LLMError(f"Gemini error {error.code}: {error}") from error
    raise LLMError("Gemini gave no answer")


def list_groq_models(get=httpx.get):
    """Model ids the Groq key can use, sorted."""
    _, _, api_key = settings("groq")
    url = GROQ_URL.rsplit("/chat/completions", 1)[0] + "/models"
    response = get(url, headers={"Authorization": f"Bearer {api_key}"}, timeout=REQUEST_TIMEOUT)
    if response.status_code != 200:
        raise LLMError(f"Groq error {response.status_code}: {_error_message(response)}")
    return sorted(item["id"] for item in response.json()["data"])


def generate(prompt, model=None, provider=None, post=httpx.post, sleep=time.sleep):
    """Send one prompt and return the answer text (reasoning blocks removed)."""
    chosen_provider, chosen_model, api_key = settings(provider, model)
    if chosen_provider == "groq":
        text = _groq(prompt, chosen_model, api_key, post, sleep)
    else:
        text = _gemini(prompt, chosen_model, api_key, sleep)
    return strip_thinking(text)


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
        post_calls.append(kwargs)
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item
    post.calls = post_calls
    return post


def run_local_checks():
    assert strip_thinking("<think>hmm\nmore</think>\nXin chào") == "Xin chào"
    assert strip_thinking(None) == ""
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

    saved = {name: os.environ.get(name) for name in ("GROQ_API_KEY", "LLM_PROVIDER", "LLM_MODEL", "LLM_MAX_TOKENS")}
    os.environ["GROQ_API_KEY"] = "test-key"
    os.environ.pop("LLM_PROVIDER", None)
    os.environ.pop("LLM_MODEL", None)
    try:
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
    if "--list" in sys.argv[1:]:
        print("\n".join(list_groq_models()))
        raise SystemExit
    try:
        provider, model, _ = settings()
    except RuntimeError as error:
        raise SystemExit(f"Skipping the live call: {error}")
    print(f"Live call with {provider}:{model} ...")
    print(generate("Reply with exactly one word: ok"))
