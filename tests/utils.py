import sys
import os


# Add parent directory to Python path
def set_path():
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
