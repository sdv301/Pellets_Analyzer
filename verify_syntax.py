import py_compile
import sys

files = [
    'database.py',
    'data_processor.py', 
    'gui.py',
    'main.py',
    'ml_optimizer.py',
    'ai_ml_integration.py',
    'ai_integration.py'
]

errors = []
for f in files:
    try:
        py_compile.compile(f, doraise=True)
        print(f"OK: {f}")
    except py_compile.PyCompileError as e:
        print(f"ERROR: {f} - {e}")
        errors.append(f)

if errors:
    print(f"\nFAILED: {len(errors)} files have syntax errors")
    sys.exit(1)
else:
    print(f"\nSUCCESS: All {len(files)} files compiled without syntax errors")
    sys.exit(0)
