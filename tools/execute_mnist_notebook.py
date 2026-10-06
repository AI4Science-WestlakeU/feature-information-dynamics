"""Execute the teaching notebook in a fresh kernel and retain its outputs."""
import json
import os
from pathlib import Path
import platform
import time
import socket
import sys
import argparse

import nbformat
from nbclient import NotebookClient
from jupyter_client import AsyncKernelManager

root = Path(__file__).resolve().parents[1]
path = root/"mnist_information_concepts.ipynb"
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
parser = argparse.ArgumentParser()
parser.add_argument("--measurements", type=Path, help="Replay a hash-checked full-test error bundle; do not remeasure risks")
args = parser.parse_args()
notebook = nbformat.read(path, as_version=4)
original_sources = [cell.source for cell in notebook.cells]
from feature_information_dynamics.bundle import load_bundle
measurement_path = args.measurements or root / "examples/mnist_class/measurements"
errors, metadata, samples = load_bundle(measurement_path)
assert errors.shape == (2, 33, 2, 10000)
assert metadata["subsets"] == [[], ["class"]]
assert metadata["log10_snr"] == [-5 + .25*i for i in range(33)]
assert metadata["noise"]["seeds"] == [20261004, 20261005]
checkpoint = json.loads((root / "examples/mnist_class/checkpoint.json").read_text())
assert metadata["checkpoint_sha256"] == checkpoint["sha256"]
assert [item["sample_id"] for item in samples] == [f"test:{i}" for i in range(10000)]
if args.measurements:
    for cell in notebook.cells:
        if cell.cell_type == "code" and 'load_bundle(ASSETS / "measurements")' in cell.source:
            cell.source = cell.source.replace('ASSETS / "measurements"', f"Path({str(args.measurements.resolve())!r})")

started = time.perf_counter()
# Reserve distinct OS-assigned free ports; some server images reuse cached ports.
leases = []
for candidate in range(62000, 64000):
    lease = socket.socket()
    try:
        lease.bind(("127.0.0.1", candidate))
        lease.listen(1)
    except OSError:
        lease.close()
        continue
    leases.append(lease)
    if len(leases) == 5:
        break
if len(leases) != 5:
    raise RuntimeError("Unable to reserve five free notebook kernel ports")
ports = [lease.getsockname()[1] for lease in leases]
manager = AsyncKernelManager(kernel_name="python3", transport="tcp", ip="127.0.0.1", cache_ports=False,
                             shell_port=ports[0], iopub_port=ports[1], stdin_port=ports[2],
                             hb_port=ports[3], control_port=ports[4])
manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
for lease in leases:
    lease.close()
NotebookClient(notebook, timeout=7200, kernel_name="python3", km=manager,
               resources={"metadata": {"path": str(path.parent)}}).execute(cleanup_kc=True)
for cell, source in zip(notebook.cells, original_sources):
    cell.source = source
notebook.metadata["measurement_execution"] = "hash-verified saved full-test measurements; fresh example inference"
nbformat.validate(notebook)
nbformat.write(notebook, path)
result = json.loads((root/"runs"/"mnist_quickstart"/"result.json").read_text())
error_count = sum(output.output_type == "error" for cell in notebook.cells
                  if cell.cell_type == "code" for output in cell.outputs)
images = sum("image/png" in output.get("data", {}) for cell in notebook.cells
             if cell.cell_type == "code" for output in cell.outputs)
assert error_count == 0 and images == 5
assert result["sample_count"] == 10000 and result["noise_repeat_count"] == 2
reference = json.loads((root/"examples"/"mnist_class"/"validation.json").read_text())
assert result["checkpoint_sha256"] == reference["checkpoint_sha256"]
assert abs(result["estimate_nats"] - reference["estimate_nats"]) < 1e-5
report = {"notebook": path.name, "execution_seconds": time.perf_counter()-started,
          "code_cells": sum(c.cell_type == "code" for c in notebook.cells),
          "error_count": error_count, "embedded_figures": images,
          "execution_platform": platform.platform(), "result": result,
          "measurement_execution": notebook.metadata["measurement_execution"]}

evidence = root/"docs"/"evidence"/"mnist_quickstart_execution.json"
evidence.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
print(json.dumps({key: value for key, value in report.items() if key != "result"}))
print(f"integral={result['estimate_nats']:.6f}; relative error={result['relative_error']:.2%}; passed={result['passed']}")
