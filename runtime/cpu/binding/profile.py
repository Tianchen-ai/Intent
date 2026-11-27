"""Select a measured CPU profile only within its recorded execution scope."""
from functools import lru_cache
from pathlib import Path

@lru_cache(maxsize=32)
def private_cache_bytes(scope, level=2):
    """Minimum data or unified cache capacity at this level across the affinity mask."""
    sizes = []
    units = {"K": 1 << 10, "M": 1 << 20}
    for cpu in scope:
        root = Path(f"/sys/devices/system/cpu/cpu{cpu}/cache")
        for entry in root.glob("index*"):
            try:
                if (entry / "level").read_text().strip() != str(level):
                    continue
                if (entry / "type").read_text().strip() not in ("Unified", "Data"):
                    continue
                size = (entry / "size").read_text().strip()
                if size and size[-1] in units:
                    capacity = int(size[:-1]) * units[size[-1]]
                else:
                    capacity = int(size)
            except (OSError, ValueError):
                continue
            sizes.append(capacity)
            break
        else:
            return 0
    return min(sizes, default=0)

@lru_cache(maxsize=32)
def calibration(scope,workers,bits):
    entries = {}
    for record in Path("/proc/cpuinfo").read_text().split("\n\n"):
        info = dict(line.split(":",1) for line in record.splitlines() if ":" in line)
        info = {k.strip():v.strip() for k,v in info.items()}
        if "processor" in info:
            entries[int(info["processor"])] = info.get("model name") or (info.get("mvendorid","")+"/"+info.get("marchid",""))
    devices={entries.get(cpu) for cpu in scope}
    if devices=={"Intel(R) Xeon(R) Platinum 8470Q"} and bits==512:
        if workers==8 and len(set(scope))==8:
            return "x86"
        if workers==1 and len(set(scope))==1:
            return "x86-1c"
    if len(devices) == 1 and set(scope) and 1 <= workers <= len(set(scope)):
        return {
            "0x5b7/0x80000000090c0d00": "sg2044",
            "0x710/0x8000000058000002": "rvv-v100",
            "Spacemit(R) X60": "k1",
            "Spacemit(R) X100": "k3-x100",
            "Spacemit(R) A100": "k3-a100",
        }.get(next(iter(devices)), "")
    return ""
