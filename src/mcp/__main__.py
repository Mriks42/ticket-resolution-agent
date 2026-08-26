"""`python -m src.mcp --scopes customer:read,policy:read`"""

from __future__ import annotations

import sys

from src.mcp.server import main

if __name__ == "__main__":
    sys.exit(main())
