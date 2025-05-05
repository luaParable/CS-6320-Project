import glob
import json
import pytest
import re

CALLSIGN = re.compile(r"[A-Z]{2,3}\d{1,4}")


@pytest.mark.parametrize("f", glob.glob("data/dev/*.json"))
def test_callsign_presence(f):
    txt = json.load(open(f))["text"].upper()
    assert CALLSIGN.search(txt), "No callsign detected"
