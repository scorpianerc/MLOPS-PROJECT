"""
Initialization file for src package
"""

from pathlib import Path
import sys

# Add src directory to path
src_dir = Path(__file__).parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

__version__ = '1.0.0'
__author__ = 'Scorpian & Rafael'
