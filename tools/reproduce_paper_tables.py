#!/usr/bin/env python3
"""Regenerate the paper tables from this repository's anonymous observations."""
from pathlib import Path
import sys

package = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(package))
from grape_auto_labeling.reproduction.tables import main


if __name__ == "__main__":
    main()
