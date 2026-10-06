# Saved full-test MNIST measurements

These are real measurements from our selected 100000-update shared U-Net,
not synthetic fixtures. The original verified bundle was recovered locally
from the full-test confirmation run; errors and metadata were copied unchanged.
The bundle manifest verifies all three files before loading.

Axes: [2 conditioning modes, 33 log10-SNR points, 2 noise draws, 10000 test images].
Each entry is a 784-pixel squared-error sum. Modes are unconditional, then
class-conditional; the ordered sample manifest records test IDs and labels.
Checkpoint identity must match ../checkpoint.json.

The gentle notebook loads these errors by default and reruns illustrative image
predictions. Full remeasurement is optional. Raw signed risk differences are
preserved; only teaching-display density and its integral are clipped.

See ../../../docs/validation.md for accuracy, provenance and limitations.
These local artifacts have not been publicly released.
