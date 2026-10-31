import traceback

print("1. Testing numpy:")
try:
    import numpy as np
    print("NumPy OK:", np.__version__)
except Exception as e:
    traceback.print_exc()

print("2. Testing pandas submodules:")
for sub in ["pandas.compat", "pandas._libs", "pandas.core.dtypes", "pandas.core.arrays"]:
    try:
        __import__(sub)
        print(f"  {sub}: OK")
    except Exception as e:
        print(f"  {sub}: EXCEPTION: {e}")
