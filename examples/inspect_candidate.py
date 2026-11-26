"""Run from the Intent checkout with the optional compilerinfra agent installed."""
import json
from pathlib import Path
import subprocess

import intentdsl

with subprocess.Popen(
    ["compilerinfra-dev", "--library", str(intentdsl.library_path()), "--workspace", ".",
     "--output-dir", "../candidate-output"],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
) as session:
    def request(action, **arguments):
        session.stdin.write(json.dumps(dict(action=action, arguments=arguments)) + "\n")
        session.stdin.flush()
        response = json.loads(session.stdout.readline())
        if "error" in response:
            raise RuntimeError(response["error"])
        return response["result"]

    try:
        candidate = request("derive", configuration={"reuse": "native"})
        built = request("compile", compiler=candidate, input=str(Path("doc/softmax.py")),
                        exit="triton", through="kir@canonical", author_input=True)
        events = request("explain", compilation_id=built["compilation_id"])
        print(built["compilation_id"], len(events["events"]), "trace entries")
    finally:
        session.stdin.close()
