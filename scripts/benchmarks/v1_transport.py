"""Opt-in legacy/V1 transport timing on an enrolled development body.

Uses the unchanged local runtime with synthetic JPEGs, real deployed backend
verification and explicit synthetic PROMOTE fixtures. This is not physical
camera/policy acceptance or an LLM benchmark. Stop the normal client first.
The legacy key arrives only through an owner-only file and is never printed.
"""
from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen

from home_cortex_client.backend import BackendSession
from home_cortex_client.commands import fulfill_pending
from home_cortex_client.credentials import Credentials, private_file
from home_cortex_client.evidence import evidence_id_for
from home_cortex_client.runtime import EdgeRuntime
from home_cortex_client.sources import SyntheticCameraSource
from home_cortex_client.stream import StreamConfig
from home_cortex_client.v1 import V1Session


class PacedSynthetic(SyntheticCameraSource):
    _next_sample: float | None = None

    def read(self):
        if self._next_sample is None:
            self._next_sample = time.monotonic()
        self._next_sample += 0.1
        time.sleep(max(0, self._next_sample - time.monotonic()))
        return super().read()


def run(args: argparse.Namespace) -> dict:
    credentials = Credentials.load(args.credentials_dir, args.body)
    key = private_file(args.legacy_key_file).read_text().strip()
    clock = lambda: datetime.now(UTC)
    names = ["vision.observe", "vision.observe_clip", "vision.autonomous_promotion"]
    results = {}

    def application_request(operation: str):
        value: dict[str, str | float] = {"operation": operation}
        if operation == "vision.observe_clip":
            value["duration_seconds"] = 1
        req = Request(args.household_origin + f"/v1/embodiments/{args.body}/observations",
                      data=json.dumps(value).encode(), headers={"Authorization": "Bearer " + key,
                      "Content-Type": "application/json"}, method="POST")
        with urlopen(req, timeout=30) as reply:
            result = json.load(reply)
        if not result.get("verified"):
            raise RuntimeError("Backend did not verify benchmark evidence")
        return result

    with tempfile.TemporaryDirectory(prefix="hc-v1-timing-") as temp, ThreadPoolExecutor(max_workers=1) as pool:
        root = Path(temp)
        source = PacedSynthetic(camera_id="camera:transport_benchmark", clock=clock, fps=10, width=1, height=1)
        runtime = EdgeRuntime(source, config=StreamConfig(port=0), clock=clock, evidence_dir=root / "media", embodiment_id=args.body)
        runtime.start()
        try:
            time.sleep(2)
            for mode in ("legacy", "v1"):
                client = BackendSession(args.household_origin, key, args.body) if mode == "legacy" else V1Session(credentials, root / "protocol")
                values = {name: [] for name in ("registration", "vision.observe", "vision.observe_clip", "promotion")}
                profile = {"revision": 1, "capabilities": [{"name": n, "schema_version": 1, "availability": "AVAILABLE"} for n in names]}
                for _ in range(args.samples):
                    start = time.perf_counter()
                    if isinstance(client, BackendSession):
                        client.connect(names)
                    else:
                        client.connect(profile)
                    values["registration"].append((time.perf_counter() - start) * 1000)
                    if isinstance(client, BackendSession):
                        client.disconnect()
                    else:
                        # Keep the durable journal open for the timed session.
                        client._control("session.disconnect", {})
                if isinstance(client, BackendSession):
                    client.connect(names)
                else:
                    client.connect(profile)
                if isinstance(client, V1Session):
                    client.start_maintenance(lambda profile=profile: profile)
                try:
                    for operation in ("vision.observe", "vision.observe_clip"):
                        for _ in range(args.samples):
                            start = time.perf_counter()
                            pending = pool.submit(application_request, operation)
                            while not pending.done():
                                if isinstance(client, BackendSession):
                                    fulfill_pending(client, runtime.packager)
                                else:
                                    client.fulfill_pending(runtime.packager)
                                time.sleep(0.01)
                            pending.result()
                            values[operation].append((time.perf_counter() - start) * 1000)
                    for _ in range(args.samples):
                        runtime.wait_for_newer_frame()
                        selected = runtime.packager.get_v1_clip(1000)
                        manifest = dict(selected.manifest, reason="visual_change")
                        manifest["evidence_id"] = evidence_id_for(manifest)
                        import base64
                        fixture = {"schema_version": 1, "embodiment_id": args.body, "operation": "vision.evidence.publish",
                                   "evidence": manifest, "media_base64": base64.b64encode(selected.payload).decode(),
                                   "promotion": {"decision": "PROMOTE", "reason": "transport_benchmark", "policy_version": "benchmark-v1"}}
                        start = time.perf_counter()
                        client.publish_evidence(fixture)
                        values["promotion"].append((time.perf_counter() - start) * 1000)
                    results[mode] = {name: {"median_ms": round(statistics.median(samples), 3),
                        "min_ms": round(min(samples), 3), "max_ms": round(max(samples), 3), "samples_ms": samples}
                        for name, samples in values.items()}
                finally:
                    client.disconnect()
        finally:
            runtime.stop()
    return {"measured_at": clock().isoformat(), "source": "synthetic JPEG / unchanged EdgeRuntime",
            "backend": args.household_origin, "device_origin": credentials.server_endpoint,
            "samples_per_operation": args.samples, "poll_interval_seconds": 0.01,
            "clip_duration_ms": 1000, "promotion": "explicit synthetic fixture; no automatic policy acceptance",
            "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--credentials-dir", type=Path, required=True)
    parser.add_argument("--legacy-key-file", type=Path, required=True)
    parser.add_argument("--household-origin", default="http://home-cortex-0")
    parser.add_argument("--body", default="embodiment:macbook-0")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.samples <= 20:
        parser.error("samples must be between 1 and 20")
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({mode: {name: stats["median_ms"] for name, stats in operations.items()}
                      for mode, operations in result["results"].items()}))


if __name__ == "__main__":
    main()
