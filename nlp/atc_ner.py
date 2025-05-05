"""
Rule-based NER for ATC phraseology.
Detects:
  • CALLSIGN  – aircraft identifiers
  • ATC_UNIT  – controller / facility identifiers
"""
import re
from typing import List, Tuple

import spacy
from spacy.language import Language
from spacy.util import filter_spans

# ---------------------------------------------------------------------- #
NUMBER_WORDS = (
    "ZERO|OH|ONE|TWO|THREE|TREE|FOUR|FIVE|FIFE|SIX|SEVEN|EIGHT|NINER|NINE"
)
TELEPHONY = r"[A-Z]+(?:\s[A-Z]+)*"
POSITION_KEYWORDS = (
    "APPROACH|ARRIVAL|DEPARTURE|DEP|TOWER|GROUND|CENTER|CTR|CONTROL|RADAR|"
    "FINALS|FLOW|TRACON|RAPCON|CLEARANCE\s+DELIVERY|DELIVERY"
)

CALLSIGN_RE = re.compile(
    rf"""
    \b(
        (?:[A-Z]{{2,3}}\d{{1,4}}(?:\s?(?:HEAVY|SUPER|LIGHT))?)        # AA564
      | (?:{TELEPHONY}\s(?:\d+|{NUMBER_WORDS})(?:\s(?:\d+|{NUMBER_WORDS})){{0,3}}
         (?:\s?(?:HEAVY|SUPER|LIGHT))?)                               # AMERICAN 564
      | (?:N\d{{1,5}}[A-Z]{{0,2}})                                    # N123AB
      | (?:[A-Z]{{1,2}}-[A-Z0-9]{{3,5}})                              # C-GABC
    )\b
    """,
    re.VERBOSE | re.IGNORECASE,
)

CONTROLLER_RE = re.compile(
    rf"""
    \b([A-Z]+(?:\s[A-Z]+)*?)\s(?:{POSITION_KEYWORDS})(?:\s\d{{1,2}})?\b
    """,
    re.VERBOSE | re.IGNORECASE,
)

# ---------------------------------------------------------------------- #
nlp = spacy.blank("en")


@Language.component("atc_regex_ner")
def atc_regex_ner(doc):
    spans = []
    for m in CALLSIGN_RE.finditer(doc.text.upper()):
        span = doc.char_span(m.start(), m.end(), label="CALLSIGN")
        if span:
            spans.append(span)
    for m in CONTROLLER_RE.finditer(doc.text.upper()):
        span = doc.char_span(m.start(), m.end(), label="ATC_UNIT")
        if span:
            spans.append(span)

    doc.ents = filter_spans(spans)
    return doc


nlp.add_pipe("atc_regex_ner", last=True)


def annotate(text: str) -> List[Tuple[str, str]]:
    """Return [(entity_text, entity_label), …]."""
    doc = nlp(text)
    return [(e.text, e.label_) for e in doc.ents]
