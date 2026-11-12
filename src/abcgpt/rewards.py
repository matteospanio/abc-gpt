"""Reward functions for GRPO on ABC tunes.

TRL calls each function with the generated `completions` (strings) and the `prompts`, among other
keyword arguments, and expects one float per completion. They are listed in `configs/grpo.yaml`
as dotted paths (`abcgpt.rewards.syntax`) and weighted with `reward_weights`.

- `syntax`: 1 if music21 parses the tune without warnings and finds notes, 0 otherwise.
- `meter`: share of bars whose length matches the time signature.
- `note_order`: share of notes written in the canonical ABC order
  (accidental, pitch, octave marks, length), e.g. `^c'2` and not `c^2` or `2c`.
"""

import contextlib
import io
import os
import re
import warnings
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction

from music21 import abcFormat, converter, meter as m21meter

_POOL = None  # created on first use: parsing with music21 is the slow part


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

def _parses(abc: str) -> float:
    try:
        with quiet() as err:
            score = converter.parse(abc, format="abc")
    except Exception:  # music21 raises many different exception types on malformed ABC
        return 0.0
    # on symbols it cannot read music21 warns and guesses (a C note): count that as an error
    return 1.0 if len(score.flatten().notes) > 0 and "WARNING" not in err.getvalue() else 0.0


def syntax(completions, prompts, **kwargs) -> list[float]:
    tunes = [tune(p, c) for p, c in zip(prompts, completions)]
    return list(_pool().map(_parses, tunes, chunksize=4))


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
