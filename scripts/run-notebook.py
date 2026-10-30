"""Execute every Jupyter cell and save its outputs into the notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient


notebook_path = Path("Loan_Default_Prediction.ipynb")
notebook = nbformat.read(notebook_path, as_version=4)
executed = NotebookClient(
    notebook,
    timeout=900,
    kernel_name="python3",
    resources={"metadata": {"path": str(Path.cwd())}},
).execute()
nbformat.validate(executed)
code_cells = [cell for cell in executed.cells if cell.cell_type == "code"]
errors = [output for cell in code_cells for output in cell.outputs if output.output_type == "error"]
if errors:
    raise RuntimeError(f"Notebook produced {len(errors)} error outputs")
nbformat.write(executed, notebook_path)
graphs = sum(
    "image/png" in output.get("data", {})
    for cell in code_cells
    for output in cell.outputs
)
print(f"Saved {len(code_cells)} executed code cells and {graphs} graphs to {notebook_path}", flush=True)
