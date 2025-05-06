"""
nlp/atc_ner.py  –  Token-level ATC NER (low FP version)
"""

from typing import List, Tuple
import re
import spacy
from spacy.matcher import Matcher, PhraseMatcher
from spacy.language import Language
from spacy.util import filter_spans
from spacy.tokens import Span

nlp = spacy.blank("en")

# ---------------- airline telephony words ---------------------------
AIRLINES = """
AMERICAN DELTA UNITED SOUTHWEST JETBLUE SPIRIT FRONTIER UPS FEDEX
LUFTHANSA RYANAIR AIR CANADA EMIRATES QANTAS AIRFRANCE BRITISH AEROMEXICO
WESTJET KOREAN JAPAN IBERIA TAP SCANDINAVIAN EASYJET WIZZ ETIHAD
TURKISH HAWAIIAN ALASKA SWISS VIRGIN AER LINGUS AEROLOGIC AIRCARGO
""".split()

DIGIT_WORD = """
ZERO ONE TWO THREE FOUR FIVE SIX SEVEN EIGHT NINER NINE TEN ELEVEN
TWELVE THIRTEEN FOURTEEN FIFTEEN SIXTEEN SEVENTEEN EIGHTEEN NINETEEN
TWENTY THIRTY FORTY FIFTY SIXTY SEVENTY EIGHTY NINETY
""".split()

# ---------------- spaCy matchers ------------------------------------
matcher = Matcher(nlp.vocab)
phrase  = PhraseMatcher(nlp.vocab, attr="LOWER")
phrase.add("AIRLINE", [nlp.make_doc(w.lower()) for w in AIRLINES])

# CALLSIGN pattern 1: ICAO prefix + 1–4 digits
matcher.add(
    "ICAO_CALL",
    [[{"TEXT": {"REGEX": "^[A-Z]{2,3}$"}}, {"IS_DIGIT": True, "OP": "+"}]],
)

# CALLSIGN pattern 2: telephony word + 1–4 digit/digit-word tokens
matcher.add(
    "TEL_CALL",
    [[{"LEMMA": {"IN": [w.lower() for w in AIRLINES]}},
      {"IS_DIGIT": True, "OP": "+"}]],
)
matcher.add(
    "TEL_CALL_WORD",
    [[{"LEMMA": {"IN": [w.lower() for w in AIRLINES]}},
      {"LOWER": {"IN": [w.lower() for w in DIGIT_WORD]}},
      {"LOWER": {"IN": [w.lower() for w in DIGIT_WORD]}, "OP": "?"}]],
)

# CALLSIGN pattern 3: tail number N123AB
matcher.add(
    "TAIL_CALL",
    [[{"TEXT": {"REGEX": r"^N\d{1,5}[A-Z]{0,2}$"}}]],
)

# HEADING
matcher.add(
    "HEADING",
    [
        # single 3-digit token (000–360)
        [{"LOWER": {"IN": ["heading", "hedding"]}},
         {"TEXT": {"REGEX": r"^[0-3]\d{2}$"}}],

        # three separate digit tokens “2 2 5”
        [{"LOWER": {"IN": ["heading", "hedding"]}},
         {"IS_DIGIT": True},
         {"IS_DIGIT": True},
         {"IS_DIGIT": True}],
    ],
)

# RUNWAY
matcher.add(
    "RUNWAY",
    [[{"LOWER": {"IN": ["runway", "rwy"]}},
      {"TEXT": {"REGEX": r"^(0[1-9]|[12]\d|3[0-6])[LRC]?$"}}]],
)


# ---------------- regex-based add-ons -------------------------------
ALTITUDE_RE      = re.compile(r"\b(\d{2,3}|ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINER)\sTHOUSAND\b")
FLIGHT_LEVEL_RE  = re.compile(r"\bFL\s?\d{2,3}\b")
FREQ_RE          = re.compile(r"\b\d{3}\.\d{1,3}\b")
ATC_UNIT_RE      = re.compile(r"\b[A-Z]+(?:\s[A-Z]+)*\s(?:APPROACH|TOWER|GROUND|CENTER|CONTROL|DEPARTURE|RADAR)\b")
AIRPORT_RE       = re.compile(r"\b[A-Z]{4}\b|\b[A-Z]{3}\b")
WAYPOINT_RE      = re.compile(r"\b[A-Z]{5}\b")
COMMAND_RE       = re.compile(r"\b(CLIMB|DESCEND|MAINTAIN|CLEARED|CONTACT|TAXI|HOLD|PROCEED|TURN|CROSS|SQUAWK|IDENT)\b", re.I)

# ---------------- pipeline component --------------------------------
@Language.component("atc_ner")
def atc_ner(doc):
    spans = []

    # token-based
    for match_id, start, end in matcher(doc):
        label = nlp.vocab.strings[match_id]
        spans.append(Span(doc, start, end, label=label))

    # regex-based
    text_up = doc.text.upper()
    def radd(lbl, rex):
        for m in rex.finditer(text_up):
            s = doc.char_span(m.start(), m.end(), label=lbl)
            if s: spans.append(s)

    radd("ALTITUDE",      ALTITUDE_RE)
    radd("FLIGHT_LEVEL",  FLIGHT_LEVEL_RE)
    radd("FREQUENCY",     FREQ_RE)
    radd("ATC_UNIT",      ATC_UNIT_RE)
    radd("AIRPORT",       AIRPORT_RE)
    radd("WAYPOINT",      WAYPOINT_RE)
    radd("COMMAND",       COMMAND_RE)

    doc.ents = filter_spans(spans)
    return doc

nlp.add_pipe("atc_ner", last=True)

def annotate(text: str) -> List[Tuple[str, str]]:
    return [(e.text, e.label_) for e in nlp(text).ents]

if __name__ == "__main__":
    s = "DENVER APPROACH AMERICAN 564 TURN RIGHT HEADING 225 DESCEND ONE THREE THOUSAND " \
        "CLEARED TO LAND RUNWAY 21L CONTACT TOWER 118.3"
    for t,l in annotate(s.upper()): print(l,":",t)