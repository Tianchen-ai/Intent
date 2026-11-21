"""Describe the actual published program and its selected parameter row on request."""
import hashlib


def selection(artifact, winner, trials_us):
    encoded = artifact.emitted
    identity = {}
    if artifact.receipt is not None:
        header = artifact.receipt
        identity = {key: header[key] for key in ("compilation_id", "compiler_candidate")}
        if "package_sha256" in header:
            identity["package_sha256"] = header["package_sha256"]
        identity["structures"] = {d["family"]: d["structure"] for d in header["decisions"]}
        identity["row_id"] = winner
    return {
        "runtime_identity": identity,
        "winner_row_id": winner,
        "artifact_sha256": hashlib.sha256(encoded).hexdigest(),
        "artifact_format": "emitted-program-json",
        "trials_us": [{"row_id": i, "samples": list(samples)}
                      for i, samples in trials_us.items()],
    }
