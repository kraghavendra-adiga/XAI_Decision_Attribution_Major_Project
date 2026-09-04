"""
explainer.py — backwards compatibility shim.
All logic has moved to:
  - image_explainer.py
  - audio_explainer.py
  - attribution.py

utils.py is re-exported as well, because _image_to_b64, _extract_keywords and
_get_nltk_stopwords used to live in this module and callers may still import
them from here.
"""

try:
    from .utils import *  # noqa: F401,F403
    from .image_explainer import *  # noqa: F401,F403
    from .audio_explainer import *  # noqa: F401,F403
    from .attribution import *  # noqa: F401,F403
except ImportError:  # pragma: no cover - fallback for direct script execution
    from utils import *  # noqa: F401,F403
    from image_explainer import *  # noqa: F401,F403
    from audio_explainer import *  # noqa: F401,F403
    from attribution import *  # noqa: F401,F403
