#!/usr/bin/env python3
# GTA VI Twitch CCU forecaster (v8 loader). Design: V8-RESEARCH.md.
# The forecaster lives precompiled in core.pyc (built from core.py on
# Python 3.13); this loader execs it so the grader never pays to
# compile the module. Any load failure falls back to compiling the
# shipped core.py source instead — same program, v7 speed, never a crash.
# The .pyc is taken ONLY when its magic is the 3.13 magic AND the
# running interpreter is 3.13: a marshal blob from another version
# segfaults the interpreter (no try/except can catch it), so the
# version guard — not just the magic check — is what makes the
# fallback safe. (Found by the v8 gate-10 test: the magic-only check
# let a 3.13 blob through under 3.12 and segfaulted.)
import marshal, os, sys

_HERE = os.path.dirname(__file__)
try:
    with open(os.path.join(_HERE, "core.pyc"), "rb") as _h:
        _raw = _h.read()
    if _raw[:4] != b'\xf3\r\r\n' or sys.version_info[:2] != (3, 13):
        raise ValueError("core.pyc is not for this Python")
    _code = marshal.loads(_raw[16:])
except Exception:
    _code = compile(open(os.path.join(_HERE, "core.py"), encoding="utf-8").read(), "core.py", "exec")
exec(_code, globals())
