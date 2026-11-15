"""Fixed Vietnamese terms shared by the translation and block-fitting prompts.

If the job folder (see workdir.py) has a glossary.json, a JSON list of strings,
that list is used instead of the default one. dub.py creates an empty one for
every new video; edit it before the translation stage to pin names and terms.

An entry may carry its meaning in parentheses, e.g. "Fe (cảm xúc hướng ngoại)":
the term is "Fe" and the meaning is explained once in words, never read out as
a parenthesis every time the term comes up.

An entry may also say what it stands for in the original speech, with "=>":

    "黄晓明 => Hoàng Hiểu Minh"          a Chinese name gets its Sino-Vietnamese reading
    "酷盖 => người ngầu"                  a fixed translation of a word
    "吴亦凡|Chris Wu => Ngô Diệc Phàm"    several spellings of the same thing, separated by |

Such an entry is only given to the LLM for the passages whose original text
contains one of its sources (so 态度 can no longer turn into another entry's
term), and the sources are also given to Whisper as hotwords. A plain entry in
Latin letters (MBTI, Fe) is likewise only used where it occurs in the original.
A plain Vietnamese entry has no source to look for, so it is always given; the
prompt tells the LLM to use it only where the speaker really means it.
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
    entries = [term.strip() for term in terms if term.strip()]
    for entry in entries:
        if "=>" in entry:
            sources, _, target = entry.partition("=>")
            if not [part for part in sources.split("|") if part.strip()] or not target.strip():
                raise ValueError(f'{path.name}: "{entry}" needs a source and a Vietnamese term on both sides of =>')
    return entries


def parse_entry(entry):
    """'黄晓明 => Hoàng Hiểu Minh' -> (['黄晓明'], 'Hoàng Hiểu Minh', None);
    'Fe (cảm xúc)' -> ([], 'Fe', 'cảm xúc'); 'MBTI' -> ([], 'MBTI', None)."""
    sources, arrow, target = entry.partition("=>")
    if not arrow:
        sources, target = "", entry
    term, meaning = split_entry(target)
    return [part.strip() for part in sources.split("|") if part.strip()], term, meaning


def split_entry(entry):
    """'Fe (cảm xúc hướng ngoại)' -> ('Fe', 'cảm xúc hướng ngoại'); 'MBTI' -> ('MBTI', None)."""
    match = _WITH_MEANING.match(entry.strip())
    if match:
        return match["term"].strip(), match["meaning"].strip()
    return entry.strip(), None


def _occurs(text, needle):
    return needle.lower() in text.lower()


def applies(entry, source_text):
    """Whether `entry` belongs in the prompt for a passage whose original text is `source_text`.

    With no source text (None) every entry applies.
    """
    if source_text is None:
        return True
    sources, term, _ = parse_entry(entry)
    if sources:
        return any(_occurs(source_text, source) for source in sources)
    if _LATIN_TERM.match(term):
        return _occurs(source_text, term)
    return True  # a Vietnamese term has nothing to look for in the Chinese


def relevant(glossary, source_text):
    return [entry for entry in glossary if applies(entry, source_text)]


def prompt_lines(glossary, source_text=None):
    """Glossary as prompt lines that never ask for a parenthesis to be repeated.

    With `source_text`, only the entries that apply to that passage (see applies()).
    """
    lines = []
    for entry in relevant(glossary, source_text):
        sources, term, meaning = parse_entry(entry)
        if sources:
            heard = " or ".join(f'"{source}"' for source in sources)
            line = f'  - {heard} must be written as "{term}"'
            if meaning:
                line += (f' (it means "{meaning}": explain it in words the first time, e.g. '
                         f'"{term}, tức là {meaning}", afterwards just say {term})')
            lines.append(line)
        elif meaning:
            lines.append(f'  - {term}: means "{meaning}". Explain it in words the first time '
                         f'(e.g. "{term}, tức là {meaning}"), afterwards just say {term}.')
        else:
            lines.append(f"  - {term}")
    return "\n".join(lines)


def whisper_hotwords(glossary):
    """What speech recognition should expect: the sources of "=>" entries (Chinese names
    and words) and Latin-script terms (MBTI, INFJ...)."""
    words = []
    for entry in glossary:
        sources, term, _ = parse_entry(entry)
        words.extend(sources if sources else [term] if _LATIN_TERM.match(term) else [])
    return " ".join(dict.fromkeys(words)) or None


def run_local_checks():
    assert split_entry("Fe (cảm xúc hướng ngoại)") == ("Fe", "cảm xúc hướng ngoại")
    assert split_entry(" MBTI ") == ("MBTI", None)
    assert split_entry("Kinh Thánh") == ("Kinh Thánh", None)
    lines = prompt_lines(["MBTI", "Fe (cảm xúc hướng ngoại)"])
    assert lines.splitlines()[0] == "  - MBTI"
    assert "afterwards just say Fe" in lines and "(cảm xúc" not in lines
    assert whisper_hotwords(["ENFJ", "MBTI", "Fe (cảm xúc hướng ngoại)", "Kinh Thánh"]) == "ENFJ MBTI Fe"
    assert whisper_hotwords(["Kinh Thánh"]) is None

    pairs = ["黄晓明 => Hoàng Hiểu Minh", "吴亦凡|Chris Wu => Ngô Diệc Phàm", "态度 => thái độ",
             "Fe (cảm xúc hướng ngoại)", "khí chất", "MBTI"]
    assert parse_entry(pairs[1]) == (["吴亦凡", "Chris Wu"], "Ngô Diệc Phàm", None)
    assert parse_entry("Fe (cảm xúc)") == ([], "Fe", "cảm xúc") and parse_entry("khí chất") == ([], "khí chất", None)
    assert parse_entry("酷盖 => cool guy (người ngầu)") == (["酷盖"], "cool guy", "người ngầu")
    text = "你想想黄晓明和chris wu, 你怎么看态度"
    assert relevant(pairs, text) == [pairs[0], pairs[1], pairs[2], "khí chất"]
    assert relevant(pairs, "我说MBTI和fe") == ["Fe (cảm xúc hướng ngoại)", "khí chất", "MBTI"]
    assert relevant(pairs, "没有任何名字") == ["khí chất"]
    assert relevant(pairs, None) == pairs
    lines = prompt_lines(pairs, "黄晓明")
    assert '"黄晓明" must be written as "Hoàng Hiểu Minh"' in lines and "thái độ" not in lines
    assert "khí chất" in lines and "Ngô Diệc Phàm" not in lines
    assert whisper_hotwords(pairs) == "黄晓明 吴亦凡 Chris Wu 态度 Fe MBTI".replace("Chris Wu", "Chris Wu")
    assert prompt_lines([], "x") == ""


GLOSSARY = load_glossary()
