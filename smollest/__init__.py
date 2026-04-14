from __future__ import annotations

import sys

import mvlm
from mvlm import anthropic, openai
from mvlm.results import report
from mvlm.web import show

__version__ = mvlm.__version__

__all__ = ["openai", "anthropic", "report", "show", "__version__"]

sys.modules.setdefault("smollest.openai", openai)
sys.modules.setdefault("smollest.anthropic", anthropic)
sys.modules.setdefault("smollest.results", sys.modules["mvlm.results"])
sys.modules.setdefault("smollest.web", sys.modules["mvlm.web"])
sys.modules.setdefault("smollest.compare", sys.modules["mvlm.compare"])
sys.modules.setdefault("smollest.defaults", sys.modules["mvlm.defaults"])
sys.modules.setdefault("smollest.candidates", sys.modules["mvlm.candidates"])
sys.modules.setdefault("smollest.cli", sys.modules["mvlm.cli"])
