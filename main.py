"""miniRT: python main.py <файл.rt> [--width W] [--height H] [--out файл.png] [--no-window]."""

import sys

from minirt.cli import main

if __name__ == "__main__":
    sys.exit(main())
