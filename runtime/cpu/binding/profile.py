"""Select a measured CPU profile only within its recorded execution scope."""
from functools import lru_cache
from pathlib import Path

@lru_cache(maxsize=32)
def calibration(scope,workers,bits):
    entries = {}
    for record in Path("/proc/cpuinfo").read_text().split("\n\n"):
        info = dict(line.split(":",1) for line in record.splitlines() if ":" in line)
        info = {k.strip():v.strip() for k,v in info.items()}
        if "processor" in info:
            entries[int(info["processor"])] = info.get("model name") or (info.get("mvendorid","")+"/"+info.get("marchid",""))
    devices={entries.get(cpu) for cpu in scope}
    if devices=={"Intel(R) Xeon(R) Platinum 8470Q"} and workers==8 and bits==512 and len(set(scope))==8:
        return "x86"
    if workers==1 and len(devices)==1:
        return {"0x5b7/0x80000000090c0d00":"sg2044","0x710/0x8000000058000002":"rvv-v100"}.get(next(iter(devices)),"")
    return ""
