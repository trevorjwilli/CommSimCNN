import sys
import os
from pathlib import Path


rootDir = Path(os.path.abspath(__name__)).parent.parent
sys.path.append(str(rootDir))

__version__ = '0.1.0'

