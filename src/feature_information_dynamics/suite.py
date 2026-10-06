"""Fail-closed validation and reproduction of one unified four-representation suite."""

import hashlib
import json
from pathlib import Path
import re

import numpy as np

from .analysis import analyze
from .bundle import load_bundle, sha256, write_json
from .noise import validate_noise_identity

REPRESENTATIONS = ("pixel", "sdvae", "vavae", "rae")
FEATURES = ["class", "mask", "masked_canny"]


def fingerprint(value: object) -> str:
    """Hash structured identities independently of JSON whitespace."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def require_hash(value: object) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Expected a concrete SHA256 identity")


def validate_protocol(protocol: dict) -> None:
    """Require concrete shared data, observation and estimator commitments."""
    if protocol.get("schema") != "unified-protocol-v2" or protocol.get("status") != "frozen":
        raise ValueError("Main reproduction requires a frozen unified protocol")
    shared = protocol["shared"]
    if shared["features"] != FEATURES:
        raise ValueError("Main conditions must be class, mask, masked_canny")
    for key in ("train_samples_sha256", "selection_samples_sha256", "evaluation_samples_sha256",
                "rgb_preprocessing_sha256", "condition_artifacts_sha256", "class_mapping_sha256"):
        require_hash(shared["data"][key])
    condition = shared["conditions"]
    if condition["canny"] != "raw_rgb_masked_then_gray_canny_100_200_then_nearest_center_crop_256":
        raise ValueError("Ordinary or differently preprocessed Canny is not this protocol")
    for key in ("mask_generation_sha256", "absent_condition_semantics_sha256"):
        require_hash(condition[key])
    observation = shared["observation"]
    grid = np.asarray(observation["log10_snr"], dtype=float)
    if grid.ndim != 1 or len(grid) < 2 or not np.isfinite(grid).all() or not (np.diff(grid) > 0).all():
        raise ValueError("Invalid frozen SNR grid")
    if observation["channel"] != "tX+(1-t)epsilon;gamma=(t/(1-t))^2":
        raise ValueError("Unexpected observation channel")
    validate_noise_identity(observation)
    seeds = observation["noise_seeds"]
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Need distinct frozen nonnegative noise seeds")
    estimator = shared["estimator"]
    expected = {"method": "native_conditioned_models",
                "guidance_scale": 1, "error_reduction": "sum"}
    if any(estimator.get(k) != v for k, v in expected.items()):
        raise ValueError("Estimator/initialization/selection protocol mismatch")
    if not isinstance(estimator.get("initialization"), str) or not estimator["initialization"].strip():
        raise ValueError("Declare the actual initialization policy")
    specs = protocol["representations"]
    if set(specs) != set(REPRESENTATIONS):
        raise ValueError("Need exactly four representation specifications")
    for rep, spec in specs.items():
        for key in ("native_source_sha256", "conditioning_patch_sha256", "pretrained_sha256",
                    "encoding_spec_sha256", "training_recipe_sha256"):
            require_hash(spec.get(key))
        if type(spec["target_dimension"]) is not int or spec["target_dimension"] < 1:
            raise ValueError("Invalid representation dimension")
        if type(spec["max_updates"]) is not int or spec["max_updates"] < 0:
            raise ValueError("Need a fixed nonnegative training budget")
        if spec.get("weight_state") not in ("raw", "ema"):
            raise ValueError("Declare the native raw or EMA evaluation weights")
        for key in ("training_target", "training_time_strategy"):
            if not isinstance(spec.get(key), str) or not spec[key].strip():
                raise ValueError(f"{rep}: declare the actual {key}")
        if "initialization_spec_sha256" in spec:
            require_hash(spec["initialization_spec_sha256"])


def validate_suite(suite_path: Path) -> tuple[dict, dict[str, Path]]:
    """Check every representation before generating any main figure.

    Hashes and declared protocol consistency establish artifact identities, not
    Bayes optimality or independent proof that execution followed the metadata.
    """
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    if suite.get("schema") != "unified-suite-v1":
        raise ValueError("Historical aggregates are not a unified main-result suite")
    protocol = suite["protocol"]
    validate_protocol(protocol)
    identity = fingerprint(protocol)
    if suite["protocol_sha256"] != identity:
        raise ValueError("Protocol fingerprint differs")
    if set(suite["bundles"]) != set(REPRESENTATIONS):
        raise ValueError("All four representations are required")
    shared, specs = protocol["shared"], protocol["representations"]
    expected_samples = None
    paths = {}
    for rep in REPRESENTATIONS:
        entry = suite["bundles"][rep]
        path = (suite_path.parent/entry["path"]).resolve()
        if not path.is_relative_to(suite_path.parent.resolve()):
            raise ValueError("Bundle must reside within the portable suite directory")
        if sha256(path/"bundle.json") != entry["bundle_sha256"]:
            raise ValueError(f"{rep}: bundle manifest identity differs")
        errors, meta, samples = load_bundle(path)
        if meta.get("protocol_sha256") != identity or meta.get("shared_protocol") != shared:
            raise ValueError(f"{rep}: shared protocol mismatch")
        if meta.get("representation_spec") != specs[rep]:
            raise ValueError(f"{rep}: representation/encoding/recipe identity mismatch")
        if meta["representation"] != rep or meta["features"] != FEATURES:
            raise ValueError(f"{rep}: feature or representation identity differs")
        if meta["log10_snr"] != shared["observation"]["log10_snr"]:
            raise ValueError(f"{rep}: SNR grid differs")
        if errors.shape[2] != len(shared["observation"]["noise_seeds"]):
            raise ValueError(f"{rep}: noise repetitions differ")
        if meta["target_dimension"] != specs[rep]["target_dimension"]:
            raise ValueError(f"{rep}: error dimension differs")
        if fingerprint(samples) != shared["data"]["evaluation_samples_sha256"]:
            raise ValueError(f"{rep}: sample manifest differs")
        for sample in samples:
            for key in ("rgb_sha256", "mask_sha256", "masked_canny_sha256"):
                require_hash(sample[key])
            if "class_id" not in sample:
                raise ValueError("Main analysis requires class identities")
        if expected_samples is not None and samples != expected_samples:
            raise ValueError(f"{rep}: images, labels or condition artifacts differ")
        expected_samples = samples
        expected = {(frozenset(subset), float(snr)) for subset in meta["subsets"] for snr in meta["log10_snr"]}
        observed = set()
        for record in meta["checkpoints"]:
            coordinate = (frozenset(record["subset"]), float(record["log10_snr"]))
            if coordinate not in expected or coordinate in observed:
                raise ValueError(f"{rep}: missing, repeated or unexpected checkpoint coordinate")
            observed.add(coordinate)
            for key in ("checkpoint_sha256", "execution_record_sha256"):
                require_hash(record[key])
            for key in ("native_source_sha256", "conditioning_patch_sha256", "encoding_spec_sha256",
                        "pretrained_sha256", "training_recipe_sha256", "weight_state",
                        "training_target", "training_time_strategy"):
                if record.get(key) != specs[rep][key]:
                    raise ValueError(f"{rep}: checkpoint {key} differs from its native specification")
            if "initialization_spec_sha256" in specs[rep] and record.get("initialization_spec_sha256") != specs[rep]["initialization_spec_sha256"]:
                raise ValueError(f"{rep}: checkpoint initialization lineage differs")
            if (type(record.get("selected_update")) is not int or
                    not 0 <= record["selected_update"] <= specs[rep]["max_updates"]):
                raise ValueError(f"{rep}: selection outside frozen budget")
        if observed != expected:
            raise ValueError(f"{rep}: incomplete eight-condition by SNR measurements")
        paths[rep] = path
    return suite, paths


def reproduce(suite_path: Path, output: Path, bootstrap: int = 1000) -> dict:
    """Compute chain and Shapley from the same validated error tensors."""
    suite, paths = validate_suite(suite_path)
    if output.exists():
        raise FileExistsError("Use a new output directory")
    if bootstrap < 20:
        raise ValueError("Need at least twenty bootstrap draws")
    output.mkdir(parents=True)
    results = {rep: analyze(path, output/rep, bootstrap) for rep, path in paths.items()}
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 2, figsize=(12, 13), constrained_layout=True)
    for row, rep in enumerate(REPRESENTATIONS):
        result = results[rep]
        for col, mode in enumerate(("chain", "shapley")):
            ax = axes[row, col]
            for i, feature in enumerate(FEATURES):
                ax.plot(result["log10_snr"], result[mode]["density_per_ln_snr"][i], label=feature)
            ax.axhline(0, color="gray", linewidth=.5)
            ax.set(title=f"{rep.upper()} — {mode}", xlabel="log10 SNR", ylabel="Signed density per ln SNR")
            ax.legend(fontsize=8)
    fig.savefig(output/"unified_chain_shapley.png", dpi=150)
    fig.savefig(output/"unified_chain_shapley.pdf")
    plt.close(fig)
    index = {"protocol_sha256": suite["protocol_sha256"], "representations": list(paths),
             "source_suite_sha256": sha256(suite_path), "same_eight_condition_tensors": True,
             "scope": "Unified declared measurement protocol; not certified Bayes MMSE"}
    write_json(output/"index.json", index)
    return index
