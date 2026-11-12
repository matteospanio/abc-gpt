"""Reward functions for GRPO on ABC tunes.

TRL calls each function with the generated `completions` (strings) and the `prompts`, among other
keyword arguments, and expects one float per completion. They are listed in `configs/grpo.yaml`
as dotted paths (`abcgpt.rewards.syntax`) and weighted with `reward_weights`.

- `syntax`: 1 if music21 parses the tune without warnings and finds notes, 0 otherwise (also if parsing
  takes more than PARSE_TIMEOUT seconds: on some malformed tunes music21 never returns).
- `meter`: share of bars whose length matches the time signature.
- `note_order`: share of notes written in the canonical ABC order
  (accidental, pitch, octave marks, length), e.g. `^c'2` and not `c^2` or `2c`.
- `structure`: how close the tune is to the control code in the prompt (sections, bars per section,
  similarity between sections), measured as IrishMAN measured it.
"""

import contextlib
import io
import os
import re
import signal
import warnings
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction

from music21 import abcFormat, converter, meter as m21meter

_POOL = None  # created on first use: parsing with music21 is the slow part
PARSE_TIMEOUT = 2  # seconds of CPU time; real IrishMAN tunes take at most 0.6 s


@contextlib.contextmanager
def quiet():
    """music21 prints warnings and debug output on malformed tunes: capture them, out of the training log."""
    warnings.filterwarnings("ignore")
    err = io.StringIO()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
        yield err


def _pool():
    global _POOL
    if _POOL is None:
        _POOL = ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 1))
    return _POOL


def tune(prompt: str, completion: str) -> str:
    """The full tune: prompt (comment line with the control code, then the header) plus completion."""
    return prompt + completion


# ---------------------------------------------------------------- syntax

class _Timeout(BaseException):  # not an Exception, so that music21's own `except Exception` cannot swallow it
    pass


def _raise_timeout(signum, frame):
    raise _Timeout


def parses(abc: str) -> float:
    """1 if music21 reads the whole tune without warnings and finds notes, within PARSE_TIMEOUT seconds."""
    # CPU time, not wall time: with all cores busy (GRPO, DataTrove) a normal tune can take seconds to finish.
    # shortcut: the timer exists only on Unix and works in the main thread (true in pool workers and DataTrove tasks)
    signal.signal(signal.SIGVTALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_VIRTUAL, PARSE_TIMEOUT)
    try:
        with quiet() as err:
            score = converter.parse(abc, format="abc")
            has_notes = len(score.flatten().notes) > 0
    except (Exception, _Timeout):  # music21 raises many different exception types on malformed ABC
        return 0.0
    finally:
        signal.setitimer(signal.ITIMER_VIRTUAL, 0)
    # on symbols it cannot read music21 warns and guesses (a C note): count that as an error
    return 1.0 if has_notes and "WARNING" not in err.getvalue() else 0.0


def syntax(completions, prompts, **kwargs) -> list[float]:
    tunes = [tune(p, c) for p, c in zip(prompts, completions)]
    return list(_pool().map(parses, tunes, chunksize=4))


# ---------------------------------------------------------------- meter

_SECTION_STYLES = {"light-light", "light-heavy", "heavy-light", "heavy-heavy", "dotted"}


def _bar_length(meter_text: str) -> Fraction | None:
    """Bar length in quarter notes from an M: field (`6/8`, `C`, `C|`); None for free meter."""
    meter_text = {"C": "4/4", "C|": "2/2"}.get(meter_text.strip(), meter_text.strip())
    try:
        return Fraction(m21meter.TimeSignature(meter_text).barDuration.quarterLength)
    except Exception:
        return None


def _is_section_bar(token) -> bool:
    """Repeat signs, first/second endings and double bars: bars next to them may be short."""
    return token.barType == "repeat" or bool(token.repeatForm) or token.barStyle in _SECTION_STYLES


def bar_lengths(abc: str) -> list[tuple[Fraction, Fraction, bool]]:
    """(written length, expected length, next to a section boundary) for each bar, as written."""
    handler = abcFormat.ABCHandler()
    handler.process(abc)
    bars, current, expected, boundary_before = [], Fraction(0), None, False
    for token in handler.tokens:
        name = type(token).__name__
        if name == "ABCMetadata" and token.tag == "M":
            expected = _bar_length(token.data)
        elif name in ("ABCNote", "ABCChord") and not token.inGrace:
            length = Fraction(token.quarterLength).limit_denominator(64)
            if token.activeTuplet is not None:  # the tokenizer does not scale tuplets: (3 means 3 in the time of 2
                length *= Fraction(token.activeTuplet.numberNotesNormal, token.activeTuplet.numberNotesActual)
            current += length
        elif name == "ABCBar":
            section = _is_section_bar(token)
            if current > 0 and expected is not None:
                bars.append((current, expected, boundary_before or section))
                current = Fraction(0)
            boundary_before = section
    if current > 0 and expected is not None:
        bars.append((current, expected, True))  # the last bar may be incomplete
    return bars


def _meter_score(abc: str) -> float:
    try:
        with quiet():
            bars = bar_lengths(abc)
    except Exception:
        return 0.0
    inner = bars[1:-1]  # the first bar may be a pickup, the last may close it
    if not inner:
        return 0.0
    ok = sum(written == expected or (written < expected and boundary) for written, expected, boundary in inner)
    return ok / len(inner)


def meter(completions, prompts, **kwargs) -> list[float]:
    return [_meter_score(tune(p, c)) for p, c in zip(prompts, completions)]


# ---------------------------------------------------------------- note order

# constructs that are not notes, removed before looking at the notes
_IGNORED = re.compile(
    r'"[^"]*"'                # chord symbols and annotations
    r"|![^!\n]*!|\+[^+\n]*\+"  # decorations
    r"|\[[A-Za-z]:[^\]]*\]"    # inline fields such as [M:3/4]
    r"|%[^\n]*"                # comments
)
_NOTE = re.compile(r"(?:\^\^|__|\^|_|=)?[A-Ga-g][,']*\d*/*\d*")  # canonical order
_OTHER = re.compile(
    r"[zxZ]\d*/*\d*"           # rests
    r"|:*\|+[:\]]*\d*|\[\d|:+"  # bar lines, repeats, endings (|1, :|2, [1)
    r"|\(\d(?::\d*){0,2}"      # tuplets
    r"|\]\d*/*\d*"             # chord close with its length
    r"|[\[\]{}()<>\-~.\s]"     # chord and grace brackets, slurs, broken rhythm, ties, spaces
    r"|[HLMOPSTuv]"            # one-letter decorations
)
_NOTE_SYMBOLS = re.compile(r"[\^_=,'0-9/A-Ga-g]+")


def _note_order_score(body: str) -> float:
    body = _IGNORED.sub(" ", body)
    good = len(_NOTE.findall(body))
    leftover = _OTHER.sub(" ", _NOTE.sub(" ", body))
    bad = len(_NOTE_SYMBOLS.findall(leftover))  # note symbols left over are out of place
    return good / (good + bad) if good + bad else 0.0


def note_order(completions, **kwargs) -> list[float]:
    return [_note_order_score(c) for c in completions]


# ---------------------------------------------------------------- structure (control code)
# Port of add_control_codes.py from TunesFormer (github.com/sander-wood/tunesformer, data_curation.zip),
# the script that wrote the control codes of IrishMAN: a tune gets the code it would have in the dataset.

_SECTION_BARS = (":|", "||", "|]", "::", "|:", "[|")
_SECTION_SPLIT = re.compile("(" + "|".join(map(re.escape, _SECTION_BARS)) + ")")
_CODE = re.compile(r"[SBE]:\d+")


def _levenshtein(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, 1):
        current = [i]
        for j, char_b in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (char_a != char_b)))
        previous = current
    return previous[-1]


def _continues_section(part: str) -> bool:
    """A part that starts with a number is a second ending (`:|2 ...`) and belongs to the previous section."""
    for char in part:
        if char in '("!':  # tuplet, annotation or decoration: a later number is not an ending
            return False
        if char.isnumeric():
            return True
        if char.isalpha():
            return False
    return False


def _sections(abc: str) -> list[str]:
    """The body split on section bar lines, each part with the bar line that closes it."""
    lines = re.split("(\n)", abc)
    lines = [lines[i * 2] + lines[i * 2 + 1] for i in range(len(lines) // 2)]
    header_lines, in_header = 0, False
    for line in lines:  # header: up to the first line that is not a field (X:, M:, K: ...) after the fields
        if len(line) > 1 and line[0].isalpha() and line[1] == ":":
            header_lines, in_header = header_lines + 1, True
        elif in_header:
            break
        else:
            header_lines += 1
    body = abc[len("".join(lines[:header_lines])):]

    parts = [part for part in _SECTION_SPLIT.split(body) if part]
    if not parts:
        return []
    if len(parts) == 1:  # no section bar line: one section, closed at its last bar line
        parts = [body[::-1].replace("|", "]|", 1)[::-1]]
    else:
        if parts[0] in _SECTION_BARS:
            parts = [parts[0] + parts[1]] + parts[2:]
        parts = [parts[i * 2] + parts[i * 2 + 1] for i in range(len(parts) // 2)]  # text + its bar line

    sections = []
    for part in parts:
        if _continues_section(part) and sections:
            sections[-1] += part
        else:
            sections.append(part)
    return sections


def control_code(abc: str) -> list[str]:
    """The control code of a tune, e.g. ["S:2", "B:8", "E:5", "B:8"]: before the bars of each section,
    its similarity (0-10) with every previous section. Empty if the tune has no body."""
    sections = _sections(abc)
    if not sections:
        return []
    code = [f"S:{len(sections)}"]
    for i, section in enumerate(sections):
        for j in range(i):
            similarity = 1 - _levenshtein(section, sections[j]) / max(len(section), len(sections[j]))
            code.append(f"E:{round(similarity * 10)}")
        for bar in _SECTION_BARS:
            section = section.strip(bar).replace(bar, "|")
        code.append(f"B:{section.count('|') + 1}")
    return code


def _unpack(code: list[str]) -> tuple[list[int], list[list[int]]]:
    """Bars of each section and, for each section, its similarities with the previous ones."""
    bars, similarities, current = [], [], []
    for token in code[1:]:
        if token[0] == "E":
            current.append(int(token[2:]))
        else:
            bars.append(int(token[2:]))
            similarities.append(current)
            current = []
    return bars, similarities


def _closeness(got: int, wanted: int, scale: int) -> float:
    return max(0.0, 1 - abs(got - wanted) / scale)


def _structure_score(requested: list[str], generated: list[str]) -> float:
    if not requested or not generated:
        return 0.0
    wanted_bars, wanted_similarities = _unpack(requested)
    bars, similarities = _unpack(generated)
    terms = [
        _closeness(len(bars), len(wanted_bars), len(wanted_bars)),
        sum(_closeness(bars[k], b, b) if k < len(bars) else 0.0 for k, b in enumerate(wanted_bars)) / len(wanted_bars),
    ]
    pairs = [(k, j, e) for k, row in enumerate(wanted_similarities) for j, e in enumerate(row)]
    if pairs:  # a single section has no similarity to match
        terms.append(sum(_closeness(similarities[k][j], e, 10) if k < len(similarities) else 0.0
                         for k, j, e in pairs) / len(pairs))
    return sum(terms) / len(terms)


def structure(completions, prompts, **kwargs) -> list[float]:
    """Mean of three closeness scores to the requested code: number of sections, bars of each section,
    similarity of each pair of sections. 1 when the tune has exactly the requested code."""
    scores = []
    for prompt, completion in zip(prompts, completions):
        comment, header = prompt.split("\n", 1)  # the prompt starts with the code as an ABC comment
        scores.append(_structure_score(_CODE.findall(comment), control_code(header + completion)))
    return scores

