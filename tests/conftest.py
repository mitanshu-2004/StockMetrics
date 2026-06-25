import os
import sys

# Put the repository root on sys.path so `import src...` resolves when pytest
# collects from the tests/ directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
