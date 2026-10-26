"""Put the project root on sys.path so tests can import `solver.*`, `app.*` and `tests.*`."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
