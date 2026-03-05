# Numba Benchmark Notes

## Install (current environment)
```powershell
cd C:\Users\kagay\Desktop\want_to_read\memorandum
python -m pip install -U numba matplotlib numpy
```

## Run benchmark + plot
```powershell
python .\benchmark_numba_only.py --sizes 96,128,160,192,224 --steps 25 --repeats 3
```

## Output
- Console table: `Pure Python` vs `Numba` runtime and speedup.
- Image file: `numba_speedup_plot.png`.

## Notes
- `Numba first-call compile time` includes JIT compile overhead.
- Compare `Numba(ms)` (after compile) to `Python(ms)` for steady-state speed.
