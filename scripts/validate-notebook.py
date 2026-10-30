"""Execute standalone notebook cells with the local CSV for project verification."""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path


notebook = json.loads(Path("Loan_Default_Prediction.ipynb").read_text())
namespace = {}
count = 0
executed = 0
compiled_plots = 0
plotting_available = importlib.util.find_spec("matplotlib") is not None
plot_cells = {3, 7, 9}
for cell in notebook["cells"]:
    if cell["cell_type"] != "code":
        continue
    count += 1
    source = "".join(cell["source"])
    compiled = compile(source, f"notebook-cell-{count}", "exec")
    if not plotting_available and count in plot_cells:
        print(f"Compiled plot cell {count}; plotting package unavailable here", flush=True)
        compiled_plots += 1
        continue
    if not plotting_available and count == 2:
        source = source.replace("import matplotlib.pyplot as plt", "")
        compiled = compile(source, f"notebook-cell-{count}", "exec")
    try:
        exec(compiled, namespace)
        executed += 1
    except Exception:
        print(f"Notebook code cell {count} failed", flush=True)
        raise
print(f"Validated {executed} executed PySpark cells and {compiled_plots} compiled plot cells", flush=True)
