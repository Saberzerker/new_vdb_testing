# check_missing_imports.py
"""Find all missing config imports - FIXED VERSION."""

import re
import sys
from pathlib import Path

# Get all Python files in src/
src_files = list(Path('src').glob('*.py'))

# Extract all config imports
config_imports = set()

for file in src_files:
    try:
        # Use UTF-8 encoding to handle special characters
        content = file.read_text(encoding='utf-8', errors='ignore')
        
        # Find: from src.config import (...)
        matches = re.finditer(r'from src\.config import \((.*?)\)', content, re.DOTALL)
        
        for match in matches:
            imports_str = match.group(1)
            # Split by comma and clean
            imports = [i.strip() for i in imports_str.split(',') if i.strip()]
            config_imports.update(imports)
        
        # Also find: from src.config import X, Y, Z
        matches = re.finditer(r'from src\.config import ([^\n(]+)', content)
        for match in matches:
            imports_str = match.group(1)
            if '(' not in imports_str:  # Not multiline
                imports = [i.strip() for i in imports_str.split(',') if i.strip()]
                config_imports.update(imports)
    
    except Exception as e:
        print(f"Warning reading {file}: {e}")

print(f"\n📊 Found {len(config_imports)} unique config imports\n")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Load actual config
try:
    import src.config as config
    actual_config = dir(config)
except Exception as e:
    print(f"❌ Error importing config: {e}")
    sys.exit(1)

# Find missing
missing = []
for imp in sorted(config_imports):
    if imp and not imp.startswith('#') and imp not in actual_config:
        missing.append(imp)

print("="*70)
if missing:
    print("❌ MISSING CONFIG VARIABLES:")
    print("="*70)
    for var in missing:
        print(f"   {var}")
    print(f"\n📝 Add these {len(missing)} variables to src/config.py")
else:
    print("✅ ALL CONFIG IMPORTS SATISFIED!")
print("="*70)