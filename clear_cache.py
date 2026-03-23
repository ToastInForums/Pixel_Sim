"""
clear_cache.py
--------------
Deletes all Numba-generated .nbi and .nbc files from __pycache__.
Regular Python .pyc bytecode is left alone.

Usage:
    python clear_cache.py            # preview what would be deleted
    python clear_cache.py --delete   # actually delete
"""
import pathlib, sys

cache_dir = pathlib.Path("D:\\VSCODE\\Personal\\Pixel_Sim\\Pixel_Sim\\__pycache__")
"""

**2. Open a terminal in VS Code** — press `` Ctrl+` `` (backtick)

**3. Run this first** to preview what will be deleted (nothing gets deleted yet):
```
python clear_cache.py
```

**4. If it looks right, run this to actually delete:**
```
python clear_cache.py --delete
```

The terminal needs to be in the same folder as `clear_cache.py` for this to work. If it's not, you can either `cd` to that folder first:
```
cd D:\VSCODE\Personal\Pixel_Sim\Pixel_Sim
```
Or just run it with the full path:
```
python D:\VSCODE\Personal\Pixel_Sim\Pixel_Sim\clear_cache.py --delete
"""

if not cache_dir.exists():
    print("No __pycache__ directory found.")
    sys.exit(0)

numba_files = list(cache_dir.glob("*.nbi")) + list(cache_dir.glob("*.nbc"))

if not numba_files:
    print("No Numba cache files found.")
    sys.exit(0)

total = sum(f.stat().st_size for f in numba_files)
print(f"Found {len(numba_files)} Numba cache files  ({total / 1024:.1f} KB)")
for f in sorted(numba_files):
    print(f"  {f.name}")

if "--delete" not in sys.argv:
    print("\nDry run — pass --delete to actually remove them.")
else:
    for f in numba_files:
        f.unlink()
    print(f"\nDeleted {len(numba_files)} files.")