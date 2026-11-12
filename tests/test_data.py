"""ABCFilter keeps real tunes and drops the ones without body, unreadable or too long."""

from datatrove.data import Document

from abcgpt.data import ABCFilter
from abcgpt.prepare import train_tokenizer

HEADER = "X:185\nL:1/8\nM:4/4\nK:Bmin\n"
BODY = "|: B3 c dBAF | B3 c dfed | B3 c dcdB |1 ABde fded :|2 ABde fdec |]\n"


def check(text, tokenizer_file, max_tokens=512):
    doc = Document(text=text, id="0", metadata={"control code": "S:2\nB:5\n"})
    return ABCFilter(tokenizer_file, max_tokens, "control code").filter(doc)


def test_abc_filter(tmp_path):
    tokenizer_file = str(tmp_path / "tokenizer.json")
    train_tokenizer([HEADER + BODY] * 10, vocab_size=300).backend_tokenizer.save(tokenizer_file)

    assert check(HEADER + BODY, tokenizer_file) is True
    assert check(HEADER, tokenizer_file) == (False, "no_body")
    assert check(HEADER + BODY.replace("dBAF", "dBjAF"), tokenizer_file) == (False, "unreadable")
    assert check(HEADER + BODY, tokenizer_file, max_tokens=20) == (False, "too_long")
