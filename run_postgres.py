#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrapper script to handle UTF-8 encoding issues with psycopg2 on Windows
"""

import sys
import os

# Force UTF-8 encoding
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    # Set locale to UTF-8
    import locale
    try:
        locale.setlocale(locale.LC_ALL, '')
    except:
        pass

# Now import the actual module
from src.storage import postgres

if __name__ == "__main__":
    postgres.main()
