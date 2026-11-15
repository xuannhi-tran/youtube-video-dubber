"""Fixed Vietnamese terms shared by the translation and block-fitting prompts.

If the job folder (see workdir.py) has a glossary.json, a JSON list of strings,
that list is used instead of the default one. dub.py creates an empty one for
every new video; edit it before the translation stage to pin names and terms.

An entry may carry its meaning in parentheses, e.g. "Fe (cảm xúc hướng ngoại)":
the term is "Fe" and the meaning is explained once in words, never read out as
a parenthesis every time the term comes up.
"""

import json
import re

from workdir import WORKDIR


DEFAULT_GLOSSARY = [
    "Kinh Thánh",
    "Cựu Ước",
    "Cơ Đốc giáo",
    "Do Thái",
    "Israel",
    "Sơn Hải Kinh",
    "Sử Ký",
]

_WITH_MEANING = re.compile(r"^(?P<term>[^()]+?)\s*\((?P<meaning>[^()]+)\)$")
_LATIN_TERM = re.compile(r"^[A-Za-z][A-Za-z0-9.+-]*$")


def load_glossary(path=WORKDIR / "glossary.json"):
    if not path.exists():
        return list(DEFAULT_GLOSSARY)
    terms = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(terms, list) or not all(isinstance(term, str) for term in terms):
        raise ValueError(f"{path.name} must be a JSON list of strings")
    return [term.strip() for term in terms if term.strip()]


def split_entry(entry):
    """'Fe (cảm xúc hướng ngoại)' -> ('Fe', 'cảm xúc hướng ngoại'); 'MBTI' -> ('MBTI', None)."""
    match = _WITH_MEANING.match(entry.strip())
    if match:
        return match["term"].strip(), match["meaning"].strip()
    return entry.strip(), None


def prompt_lines(glossary):
    """Glossary as prompt lines that never ask for a parenthesis to be repeated."""
    lines = []
    for entry in glossary:
        term, meaning = split_entry(entry)
        if meaning:
            lines.append(f'  - {term}: means "{meaning}". Explain it in words the first time '
                         f'(e.g. "{term}, tức là {meaning}"), afterwards just say {term}.')
        else:
            lines.append(f"  - {term}")
    return "\n".join(lines)


def whisper_hotwords(glossary):
    """Latin-script terms (MBTI, INFJ...) that speech recognition should expect."""
    terms = [split_entry(entry)[0] for entry in glossary]
    return " ".join(term for term in terms if _LATIN_TERM.match(term)) or None


def run_local_checks():
    assert split_entry("Fe (cảm xúc hướng ngoại)") == ("Fe", "cảm xúc hướng ngoại")
    assert split_entry(" MBTI ") == ("MBTI", None)
    assert split_entry("Kinh Thánh") == ("Kinh Thánh", None)
    lines = prompt_lines(["MBTI", "Fe (cảm xúc hướng ngoại)"])
    assert lines.splitlines()[0] == "  - MBTI"
    assert "afterwards just say Fe" in lines and "(cảm xúc" not in lines
    assert whisper_hotwords(["ENFJ", "MBTI", "Fe (cảm xúc hướng ngoại)", "Kinh Thánh"]) == "ENFJ MBTI Fe"
    assert whisper_hotwords(["Kinh Thánh"]) is None


GLOSSARY = load_glossary()
