"""The rewards must tell real tunes from broken ones."""

import re
import time

from abcgpt import rewards

# two tunes from the IrishMAN validation split: prompt (control code + header) and body
TUNES = [
    ("%S:2 B:5 E:6 B:5\nX:185\nL:1/8\nM:4/4\nK:Bmin\n",
     "|: B3 c dBAF | B3 c dfed | B3 c dcdB |1 ABde fded :|2 ABde fdec |: dfaf bfaf | dfaf g2 fg | \n"
     " afge d3 B |1 ABde fdec :|2 ABde fded |]\n"),
    ("%S:2 B:8 E:5 B:8\nX:268\nL:1/8\nQ:3/8=120\nM:6/8\nK:A\n",
     " Aed cBA | BBB B2 c | d2 f ecA | G2 A B3 | Aed cBA | BBB TB2 c | d2 e fga | gaf e3 :: eba gfe | \n"
     " fff Tf2 g | a2 g Tf2 e | d2 c TB3 | eba gfe | Tf2 g a2 f | edc BAG | AAA A3 :|\n"),
]
PROMPTS = [p for p, _ in TUNES]


def score(name, completions):
    return getattr(rewards, name)(completions=completions, prompts=PROMPTS)


def test_real_tunes_score_high():
    bodies = [c for _, c in TUNES]
    assert score("syntax", bodies) == [1.0, 1.0]
    assert score("meter", bodies) == [1.0, 1.0]
    assert score("note_order", bodies) == [1.0, 1.0]


def test_no_notes_or_unreadable_symbols_fail_syntax():
    assert score("syntax", ["|: :: |]\n", "\n"]) == [0.0, 0.0]
    assert score("syntax", [c.replace(" | ", " j | ", 1) for _, c in TUNES]) == [0.0, 0.0]  # music21 reads j as C


def test_tune_that_hangs_music21_fails_syntax_in_time():
    hang = "X:1\nL:1/4\nM:6/4\nK:D\n", 'AFd14737183{^oder |"^/"ers |({'  # minimised from a generated tune
    start = time.monotonic()
    assert rewards.syntax(completions=[hang[1]], prompts=[hang[0]]) == [0.0]
    assert time.monotonic() - start < rewards.PARSE_TIMEOUT + 2


def test_cut_bars_lower_meter():
    cut = [re.sub(r"\| (\S+) ", "| ", c, count=4) for _, c in TUNES]  # drop a group of notes in four bars
    assert all(s < 0.9 for s in score("meter", cut))


def test_length_before_pitch_lowers_note_order():
    swapped = [re.sub(r"([A-Ga-g])(\d)", r"\2\1", c) for _, c in TUNES]  # B3 -> 3B
    assert all(s < 0.95 for s in score("note_order", swapped))


def test_control_code_matches_irishman():
    for prompt, body in TUNES:
        assert " ".join(rewards.control_code(prompt + body)) == prompt.split("\n")[0][1:]


def test_structure_rewards_the_requested_code():
    bodies = [c for _, c in TUNES]
    assert score("structure", bodies) == [1.0, 1.0]
    longer = [p.replace("B:5", "B:16").replace("B:8", "B:16") for p in PROMPTS]  # ask for longer sections
    assert all(s < 1.0 for s in rewards.structure(completions=bodies, prompts=longer))
