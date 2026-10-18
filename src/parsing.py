"""Extract the final differential (JSON array of {diagnosis, probability}) from model text."""
import json
import re

_DECODER = json.JSONDecoder()


def _clean(items):
    out = []
    for d in items:
        if isinstance(d, dict) and isinstance(d.get("diagnosis"), str) and d["diagnosis"].strip():
            try:
                p = float(d.get("probability"))
            except (TypeError, ValueError):
                p = None
            out.append({"diagnosis": d["diagnosis"].strip(), "probability": p})
    return out


def _bare_objects(text):
    """Fallback: the last run of consecutive {"diagnosis": ...} objects with no enclosing [ ]."""
    runs, run, end = [], [], -1
    for m in re.finditer(r"\{", text):
        if m.start() < end:
            continue
        try:
            obj, stop = _DECODER.raw_decode(text, m.start())
        except json.JSONDecodeError:
            continue
        if not (isinstance(obj, dict) and "diagnosis" in obj):
            continue
        if run and not re.fullmatch(r"[\s,]*", text[end:m.start()]):
            runs.append(run)
            run = []
        run.append(obj)
        end = stop
    if run:
        runs.append(run)
    return runs[-1] if runs else []


def parse_dx(text):
    """Return [{diagnosis, probability}] from the LAST valid JSON array of objects in text.

    Tries every '[' from the end with raw_decode, so a malformed earlier draft cannot swallow
    the final answer (a regex span would merge the two). Falls back to a trailing run of bare
    objects, which base models often emit without the enclosing brackets.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    for i in reversed([m.start() for m in re.finditer(r"\[", text)]):
        try:
            arr, _ = _DECODER.raw_decode(text, i)
        except json.JSONDecodeError:
            continue
        if isinstance(arr, list) and (out := _clean(arr)):
            return out
    return _clean(_bare_objects(text))
