# home-cortex-client

Independent edge-device runtime for Home Cortex. This project owns camera
capture, device-local frame handling, and live-stream publication. It does not
import Home Cortex backend modules and it does not mutate the household graph.

The current implementation is the preserved Mac development client: OpenCV
captures the built-in camera and a standard-library HTTP server publishes an
MJPEG preview. A synthetic source supports hardware-free development and tests.
No detector, tracker, clip worker, observation publisher, FFmpeg process, media
gateway, Tapo integration, or MicroDuck adapter is implemented.

## Install and run

```sh
python -m pip install -e '.[camera,dev]'
home-cortex-client --source synthetic --host 127.0.0.1 --port 8088
home-cortex-client --source mac --host 127.0.0.1 --port 8088
```

To open a Home Cortex runtime session for the preconfigured MacBook body, set
`HOME_CORTEX_CLIENT_CORTEX_URL` and `HOME_CORTEX_CLIENT_CORTEX_API_KEY` in the
device environment before starting the client. The default embodiment ID is
`embodiment:macbook-0`; override it with `HOME_CORTEX_CLIENT_EMBODIMENT_ID` or
`--embodiment-id`. The client sends only that ID and current
`vision.observe` availability. Home Cortex must already contain the body and
its agent assignment; an unknown ID fails registration and stops the client.
While running, the client heartbeats every 10 seconds and sends an explicit
disconnect on clean shutdown. An abrupt process or network loss has no
server-side session expiry yet, so Home Cortex may report the old session
online until a server restart or a later successful disconnect.

The preview is served at `http://127.0.0.1:8088/live.mjpg`, the small viewer at
`/`, and runtime health at `/health`. Use `--help` for all configuration flags.
Each flag defaults from the corresponding `HOME_CORTEX_CLIENT_*` variable shown
in `.env.example`; export those variables in the device environment before
starting the process. The runtime does not load `.env` files or require backend
environment variables.

## Runtime boundary

```text
home_cortex_client  -- serialized observation/clip records -->  home_cortex
home_cortex_client  -- encoded live stream ---------------->  browser/gateway
home_cortex         -- backend API ------------------------->  home_gui
```

The implemented channels today are the encoded MJPEG stream and the optional
Home Cortex embodiment session protocol. The future
structured channel is deliberately a serialized protocol boundary: the client
will send canonical `VisualObservation` JSON/NDJSON and evidence-clip metadata
to a backend transport adapter. There is no production observation HTTP endpoint
yet, so this project does not pretend to publish observations. Backend contract
definitions remain authoritative and no Python package crosses the repository
boundary.

Replacing `MacCameraSource` with a future MicroDuck source should change device
configuration and the capture adapter only. Backend evidence, spatial, identity,
and reconciliation semantics must not change.

## Tests

```sh
python -m pytest -q
```

Tests use the synthetic source and mocks. The physical camera smoke test is
manual and opt-in:

```sh
HOME_CORTEX_CLIENT_CAMERA_SMOKE=1 python -m pytest -q -m manual
```

Vision implementation remains paused pending MicroDuck hardware.
