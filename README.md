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

Capture stays local. The client keeps a rolling buffer of short JPEG segments,
60 seconds by default (`HOME_CORTEX_CLIENT_BUFFER_SECONDS`). Nothing in that
buffer is uploaded until Home Cortex asks. A permission denial, a busy camera,
or an encoding failure leaves the process running and the session connected.
`vision.observe` is advertised only while a fresh frame is available.

Local checks do not call a model:

```sh
home-cortex-client camera status
home-cortex-client camera latest-frame
home-cortex-client camera list-buffer
home-cortex-client camera save-last 5
home-cortex-client evidence latest
home-cortex-client evidence clip --seconds 8
home-cortex-client evidence inspect evidence:...
```

Those commands talk to the debug routes on the local preview server. Selected
evidence is written under `HOME_CORTEX_CLIENT_EVIDENCE_DIR`, or a temporary
directory when that variable is unset, and removed by count and age
(`HOME_CORTEX_CLIENT_EVIDENCE_MAX_ITEMS`, default 8).
`HOME_CORTEX_CLIENT_FRESHNESS_SECONDS` is the oldest still that
`vision.observe` may call current.

With a session open, the client polls for `vision.observe` and
`vision.observe_clip` about twice a second and submits one selected still or
clip. It does not upload continuously.

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

The implemented channels are the encoded MJPEG preview, the embodiment session
(heartbeat plus an explicit still or clip when Home Cortex asks), and the local
debug routes. Detector `VisualObservation` publishing is not implemented.
Backend contract definitions remain authoritative and no Python package crosses
the repository boundary.

Replacing `MacCameraSource` with a future MicroDuck source should change device
configuration and the capture adapter only. Backend evidence, spatial, identity,
and reconciliation semantics must not change.

## Tests

```sh
python -m pytest -q
```

Tests use the synthetic source and mocks. They cover the bounded buffer,
evidence manifests, and capture-failure recovery. The physical camera smoke
test is manual and opt-in:

```sh
HOME_CORTEX_CLIENT_CAMERA_SMOKE=1 python -m pytest -q -m manual
```

Detector, tracking, and automatic clip selection remain unimplemented.

cd /Users/jiankuang/Workspace/home-cortex-client

export HOME_CORTEX_CLIENT_CORTEX_URL="http://home-cortex-0"
export HOME_CORTEX_CLIENT_CORTEX_API_KEY="$(
  ssh -o BatchMode=yes jkuang@home-cortex-0 \
    'docker exec cortex-cortex-api-1 printenv CORTEX_API_KEY'
)"

.venv/bin/home-cortex-client \
  --source mac \
  --embodiment-id embodiment:macbook-0 \
  --host 127.0.0.1 \
  --port 8088