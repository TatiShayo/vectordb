import os
from pathlib import Path

root = Path(r"c:\Users\TATI\Desktop\DEV\vectordb\vectordb")
for p in root.glob("**/*.py"):
    if ".pytest_cache" in p.parts or "venv" in p.parts:
        continue
    content = p.read_text(errors='ignore')
    if "on_event" in content:
        print(f"Found on_event in {p}")
