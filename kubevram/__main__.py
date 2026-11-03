"""
KubeVRAM module entrypoint for execution via `python3 -m kubevram`.
"""

import sys
from kubevram.cli import main

if __name__ == "__main__":
    sys.exit(main())
