"""Put ``src/`` on the import path so ``tests/`` can import the analysis modules.

With this file present, ``pytest tests/`` works from the repository root without
setting ``PYTHONPATH``. The ``PYTHONPATH=src`` invocations documented in the
README remain valid, and the standalone runner is unaffected.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
