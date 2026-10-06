# Measurement bundle v0.1

`bundle.json` lists exactly `errors.npy`, `metadata.json`, and `samples.json`, with
their SHA256 hashes and byte sizes. Loads reject incomplete or changed files.
NPY has no object arrays or pickle. Error axes are `[subset, snr, noise_repeat,
sample]`; all condition subsets occur exactly once. Sample IDs are unique and
the same axes apply to all conditions. Metadata identifies target dimension,
sum reduction, features, condition subsets, and strictly increasing log10 SNR.

The MNIST measurement records original dataset SHA256, input scaling, checkpoint
selection histories and hashes, and the precise noise seed. Noise is generated
on CPU using SHA256(`fid-noise-v1:{seed}:{sample_id}`), first 16 bytes as a
little-endian integer, NumPy PCG64, standard_normal float64 then cast float32.
The same vector is used over SNRs; different condition models receive identical
observations. Include the NumPy version when reproducing exact streams.

The prototype stores one dense tensor per bundle. Streaming shards, image crop
hashes, latent normalization, stochastic-encoder seeds and third-party identities
must be added and verified before the four-representation publication bundle.
Current MNIST schema support is not a claim that those large-image fields are
already implemented.

Historical aggregate inputs instead use `legacy-aggregate-v1`: eight mean-risk
curves per representation, with per-source byte hashes and field names. They are
not sample-level bundles, do not provide paired intervals, and cannot be passed
off as frozen unified main results. Their separate CLI makes this distinction
explicit.
