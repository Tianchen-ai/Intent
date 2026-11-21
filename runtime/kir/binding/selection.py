"""Describe the actual published program and its selected parameter row on request."""
import hashlib
import json


def selection(data, winner, trials_us):
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    return {
        "winner_row_id": winner,
        "artifact_sha256": hashlib.sha256(encoded).hexdigest(),
        "artifact_format": "emitted-program-json",
        "trials_us": [{"row_id": i, "samples": list(samples)}
                      for i, samples in trials_us.items()],
    }
