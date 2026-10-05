# home-cortex-client

Independent edge-device runtime for Home Cortex. This project owns camera
capture, device-local frame handling, and live-stream publication. It does not
import Home Cortex backend modules and it does not mutate the household graph.

The current implementation is the preserved Mac development client: OpenCV
captures the built-in camera and a standard-library HTTP server publishes an
MJPEG preview. A synthetic source supports hardware-free development and tests.
No tracker, FFmpeg process, media gateway, Tapo integration, or MicroDuck
adapter is implemented. A deterministic
frame-difference detector can retain local candidate clips.

## Install and run

```sh
python -m pip install -e '.[camera,dev]'
home-cortex-client --source synthetic --host 127.0.0.1 --port 8088
home-cortex-client --source mac --host 127.0.0.1 --port 8088
```

The Home Cortex-facing default is Client Interface V1 over TLS 1.3 with mutual
certificate authentication. Enroll the existing body before starting a connected
client; see [V1 provisioning](#v1-provisioning). The default embodiment ID remains
`embodiment:macbook-0`, independent of hostname, IP or session. Home Cortex must
already contain that body and its agent assignment. Registration never creates
or reassigns an embodiment. Without credentials or a configured backend, local
capture and preview remain available.

The server returns heartbeat and lease durations. An independent maintenance
thread renews the lease while capture/upload is busy. A conservative monotonic
watchdog clears effective capabilities on expiry; stale/replaced sessions
re-register with a new fence. Authentication/protocol failure remains visible
and stops reconnect attempts until configuration is corrected. Clean shutdown
disconnects explicitly. Home Cortex also fences idle sessions after lease expiry.

Capture stays local. The client keeps a rolling buffer of short JPEG segments,
60 seconds by default (`HOME_CORTEX_CLIENT_BUFFER_SECONDS`). Nothing in that
buffer is uploaded until Home Cortex asks. A permission denial, a busy camera,
or an encoding failure leaves the process running and the session connected.
Configured implemented capabilities remain in the V1 manifest during temporary
failure, with `TEMPORARILY_UNAVAILABLE` and a standard error reason. Revision
increments only when the manifest changes. The default configured profile is
`vision.observe`; enable clip/promotion explicitly after provisioning their grants.

The same capture loop scores a few grayscale samples per second
(`HOME_CORTEX_CLIENT_DETECTOR_HZ`, default 4). The score is the mean absolute
difference from the previous sample, divided by 255. Crossing
`HOME_CORTEX_CLIENT_MOTION_START_THRESHOLD` (default 0.08) opens a motion
interval. The interval stays open while the score remains at or above
`HOME_CORTEX_CLIENT_MOTION_CONTINUE_THRESHOLD` (default 0.04) and closes after
`HOME_CORTEX_CLIENT_SETTLING_SECONDS` (default 1) of quieter samples. One
sample that changes most of the frame is also reported as a scene change.
The score carries no household meaning.

A closed interval becomes one local candidate clip: the trigger plus
`HOME_CORTEX_CLIENT_PRE_ROLL_SECONDS` (default 3) and
`HOME_CORTEX_CLIENT_POST_ROLL_SECONDS` (default 5), cut from this same ring
buffer. The clip reuses the evidence manifest and sha256. `upload_state` is
`local_only`. The Home Cortex poll submits only an explicit `vision.observe`
or `vision.observe_clip`. Pauses shorter than
`HOME_CORTEX_CLIENT_MERGE_GAP_SECONDS` (default 2) stay in one candidate.
Motion longer than `HOME_CORTEX_CLIENT_MAXIMUM_CANDIDATE_DURATION` (default
30 seconds) is cut into successive candidates; those pieces stay separate, and
the pre-roll and post-roll are what overlap. Retention is bounded by
`HOME_CORTEX_CLIENT_MAX_CANDIDATES`,
`HOME_CORTEX_CLIENT_MAX_CANDIDATE_AGE_SECONDS`, and
`HOME_CORTEX_CLIENT_MAX_CANDIDATE_DISK_BYTES`.

On the Mac, sitting still should leave the detector idle. Waving a hand or
walking through the view should produce one motion candidate. Covering and
uncovering the camera should produce a scene-change candidate. `candidates list`
shows the score, the clip bounds, and `upload_state`.

Local checks do not call a model:

```sh
home-cortex-client camera status
home-cortex-client camera latest-frame
home-cortex-client camera list-buffer
home-cortex-client camera save-last 5
home-cortex-client evidence latest
home-cortex-client evidence clip --seconds 8
home-cortex-client evidence inspect evidence:...
home-cortex-client detector status
home-cortex-client candidates list
home-cortex-client candidates inspect candidate:...
home-cortex-client events list
home-cortex-client events show candidate:...
home-cortex-client events stats
home-cortex-client events clip candidate:... --output ./candidate-review --open
```

Those commands talk to the debug routes on the local preview server. Selected
evidence is written under `HOME_CORTEX_CLIENT_EVIDENCE_DIR`, or a temporary
directory when that variable is unset, and removed by count and age
(`HOME_CORTEX_CLIENT_EVIDENCE_MAX_ITEMS`, default 8).
`HOME_CORTEX_CLIENT_FRESHNESS_SECONDS` is the oldest still that
`vision.observe` may call current.

`events list` gives the trigger interval, duration, peak score, evidence ID,
retention status, and `local_only` transfer state. `events show` adds the
detector configuration captured when the candidate was made, raw trigger
interval, requested pre/post-roll window, actual selected clip interval,
sequence range, SHA-256, local payload path, and transfer history. `events
stats` combines detector and candidate counters, including sampled frames,
triggered/discarded/merged events, retained bytes, and expirations. The
`events clip` command checks the hash, writes `clip.hcc`, `manifest.json`, JPEG
frames, and a local `index.html` playback/scrub page. No Home Cortex upload is
involved. Candidate metadata and clips are indexed again after a client
restart; count, age, and byte limits expire old clips. The local default
preview/debug listener is `127.0.0.1:8088`; keep it loopback-only when
inspecting private footage.

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
ruff check .
pyright
```

Tests use the synthetic source and synthetic grayscale sequences. They cover
the bounded buffer, evidence manifests, capture-failure recovery, and local
visual-change candidates. The physical camera smoke test is manual and opt-in:

```sh
HOME_CORTEX_CLIENT_CAMERA_SMOKE=1 python -m pytest -q -m manual
```

## Stage 2 MacBook validation

Start from an interactive Terminal with Camera permission. A fixed evidence
directory lets the client re-index retained candidates after restart. The
backend variables are needed for the embodiment to show Online and for the
Stage 1 manual-observation regression check. Candidate creation itself stays
local.

```sh
cd /Users/jiankuang/Workspace/home-cortex-client
export HOME_CORTEX_CLIENT_EVIDENCE_DIR="$HOME/Library/Caches/home-cortex-client/evidence"

export HOME_CORTEX_CLIENT_INTERFACE=v1
# Enroll first. The HTTPS endpoint is read from the protected identity file.

.venv/bin/home-cortex-client \
  --source mac \
  --embodiment-id embodiment:macbook-0 \
  --host 127.0.0.1 \
  --port 8088
```

Session registration starts asynchronously; the startup line can say
`disconnected` while connecting. Verify MacBook is Online in Home Cortex and
check stderr for standard V1 failure codes. Keep this Terminal running.
In a second Terminal, use:

```sh
cd /Users/jiankuang/Workspace/home-cortex-client
.venv/bin/home-cortex-client camera status
.venv/bin/home-cortex-client events stats
.venv/bin/home-cortex-client events list
.venv/bin/home-cortex-client events show 'candidate:<id>'
.venv/bin/home-cortex-client events clip 'candidate:<id>' --open
```

For a live check, observe the counter and list before each scenario: (A) stay
still for 30 seconds; (B) move a hand/body and wait for 5 seconds of post-roll;
(C) move twice with a pause shorter than the 2-second merge gap; (D) wait well
beyond the merge gap, then move again; (E) keep moving beyond the 30-second
maximum trigger length; (F) make several candidates without a manual
observation request. The expected results are few events while still, one
candidate with pre/post context for a movement, one merged candidate for C,
two for D, bounded overlapping clips for E, and `local_only` for every
candidate in F. Use `events clip` to inspect the actual frames and timestamps.
This test needs a person in front of the camera; synthetic tests cannot prove
its real-world false-positive or false-negative rate.

Port 8088 is Mac-local preview and inspection. The client connects outbound to
Home Cortex on HTTPS port 8443, polls for explicit Stage 1 observation commands, and
does not send autonomous candidates. A manual `vision.observe` or
`vision.observe_clip` creates separate on-demand evidence; it does not change
candidate transfer state.

## Stage 3 semantic candidate contract

`home_cortex_client.semantic` defines a **contract only** for future local
perception. `VisualEventCandidate.from_stage2(record)` freezes the Stage 2
`candidate:` provenance: evidence and embodiment IDs, camera, detector,
configuration, mechanical scores, sequence range, clip times, and media hash.
It is distinct from Home Cortex's persistent `visual_candidate:` physical
instance identity. `SemanticCandidate` adds an analysis timestamp, model ID and
version, provisional category/activity labels with semantic confidence, a
symbolic fingerprint, and an optional advisory promotion recommendation.

V1 categories are `person`, `animal`, `vehicle`, `package`, `door`, `furniture`,
`screen`, `food_or_drink`, and `unknown_object`. Coarse activities are
`person_present`, `person_entered`, `person_left`, `object_activity`,
`large_scene_change`, and `unknown_activity`. Named identities and household
facts cannot be represented. Confidence is in `[0, 1]` and is model confidence;
physical-estimation `p95` does not appear in this contract.

The fingerprint contains sorted distinct category labels, an optional activity,
and an optional opaque SHA-256. It supports exact symbolic comparison; an
opaque hash is comparable only under a compatible model/version. It does not
imply object identity or require a vector database. A successful analysis
requires model provenance and a fingerprint consistent with its observations.
`UNAVAILABLE` and `FAILED` require a machine-readable failure code and carry
no semantic claims or promotion recommendation. Source evidence remains valid
when semantic analysis fails.

`PROMOTE`, `RETAIN_LOCAL`, and `DROP` are **recommendations**, each with a
machine-readable reason such as `transition:empty->person_present`,
`new_category:animal`, `near_duplicate`, or `low_semantic_content`. Constructing
or deserializing one does not upload evidence, delete local clips, notify anyone,
or mutate SurrealDB. Stage 2 retention still applies. A consumer must
verify the Stage 2 manifest and clip hash before successful analysis, and any
actual transfer needs a separate authorized action. The V1 mapping is stable
JSON with `schema_version: 1`; parsing rejects unknown authority-bearing fields
and unsupported versions.

## Stage 3 local semantic analyzer

`HOME_CORTEX_CLIENT_ANALYZER` selects an optional background filter. The default
is `off`, which does not load a model and leaves Stage 2 behavior unchanged.
`vision` runs Apple's human-rectangle, animal, and image classifiers through a
small helper. `hog` runs OpenCV's default people detector and can report only
`person`. Neither backend is a conversational vision-language model, and neither
is enabled until an operator chooses it.

The worker reads clips that Stage 2 already stored. It does not open a camera
and does not upload. For each clip it keeps at most four frames: the first, the
strongest consecutive luma change, the middle, and the last. Labels pass through
an explicit table onto the V1 vocabulary. Several people become one `person`
observation. Names and scene words such as `outdoor` or `sky` are discarded.
Person or animal confidence at or above 0.25 is kept. Other mapped objects need
0.50. A successful empty result means the filter saw no supported category; the
clip stays local. `person_entered` and `person_left` compare the first and last
sample. A category change without a person becomes `object_activity`. A scene
trigger, or a scene score of at least 0.45 with no category, becomes
`large_scene_change`.

Results are written next to the candidate directory, under a sibling whose name
ends in `-semantics`, as `semantic.json`, `policy.json`, and a `trace.json` of
sample indices, raw labels, and timing. The trace is not part of the semantic
contract. `semantic.json` uses the V1 contract. On `SUCCESS` its `promotion`
field carries the advisory decision and reason. On `UNAVAILABLE` or `FAILED`
that field stays unset, and `policy.json` still records the fallback.
`SUCCESS` and `FAILED` are not retried. `UNAVAILABLE` is retried after the
model becomes runnable. `model_unavailable`, `model_error`, `corrupt_evidence`,
`hash_mismatch`, and `evidence_unavailable` leave the Stage 2 payload in place.

While the client packages an explicit `vision.observe` or `vision.observe_clip`,
it sets a hold. The worker will not start another frame until that returns.
The helper process is also started at a lower scheduling priority. An analysis
already inside one frame finishes that frame. Interactive observation is not
queued behind the filter. Inspect stored results with `semantics list` and
`semantics show` against the running client. The MacBook corpus used to compare
readers lives in `evaluation/macbook-0-semantic-v1/`; its JPEG pixels are local
and its manifest records the fingerprint.

## Stage 3 promotion policy

`home_cortex_client.policy` chooses `PROMOTE`, `RETAIN_LOCAL`, or `DROP` from
a `SemanticCandidate`, recent local semantic results, and client configuration.
`PROMOTE` means the clip deserves later Home Cortex processing. It does not
establish a household fact, notify anyone, or upload bytes. The policy does
not read SurrealDB, conversation history, or household relationships.

The MacBook process is the first implementation of that contract. The decision
uses the canonical records only. It does not read a camera index, an
accelerator, or a vendor observation. Another embodiment can supply its own
`decide` function and run `conformance_failures` when it claims
`vision.autonomous_promotion`.

Local capability names are `vision.observe`, `vision.observe_clip`,
`vision.semantic_filter`, and `vision.autonomous_promotion`. Each is a boolean
the client declares from what it can currently do. The embodiment id does not
imply them. The V1 manifest includes implemented names in
`HOME_CORTEX_CLIENT_CAPABILITIES`, with their current availability. The default
is `vision.observe`. Add `vision.observe_clip` or
`vision.autonomous_promotion` only when the embodiment record already lists
that name. `GET /debug/capabilities` shows both the local map and that session
list.

With an analyzer selected, a change from an empty recent window to
`person_present`, `animal_present`, or `mixed_activity` is promoted. A new
supported category and a change between known coarse states are promoted.
The same non-empty state inside the window (8 seconds by default) is
`RETAIN_LOCAL` with reason `near_duplicate` or `repetition:<state>`. Set
`HOME_CORTEX_CLIENT_PROMOTION_REPEAT=drop` to record `DROP` instead; the clip
stays on disk either way. An empty scene stays `RETAIN_LOCAL` with
`low_semantic_content`. Confidence below
`HOME_CORTEX_CLIENT_PROMOTION_MIN_CONFIDENCE` (default 0.5) keeps a candidate
that would otherwise be promoted. That number is an uncalibrated gate for this
client, not a probability shared with any other perception stack.

If semantic analysis is `UNAVAILABLE` or `FAILED`, the policy records
`RETAIN_LOCAL` with `semantic_unavailable` or `semantic_failed` and leaves the
Stage 2 evidence in place. `HOME_CORTEX_CLIENT_PROMOTION_OVERRIDE` may be
`force_promote` or `force_retain`. Those apply before the learned rules, do
not invent observations, and do not upload. The analyzer default remains
`off`, so this policy does not run until an operator selects a model.

## Selective evidence promotion

`home_cortex_client.publish` sends a `PROMOTE` decision on the same session
the client already uses for explicit observation. The body is
`vision.evidence.publish`: schema version `1` as an integer, the existing
evidence manifest, the media bytes, the promotion decision, and an optional
semantic candidate. `RETAIN_LOCAL` and `DROP` stay on this machine. Explicit
`vision.observe` and `vision.observe_clip` requests are a separate pull and
are still answered from the local buffer.

While Home Cortex is unreachable, promoted items sit in a bounded queue
(8 by default). The oldest queued item is dropped from the queue when that
bound is exceeded. The clip file stays. A later flush retries. The same
evidence id is not uploaded twice. The camera and the analyzer keep running.

`schema_version`, the evidence id, capture times, media type, and content
hash travel with the upload. `client_runtime_version`, `perception_model_id`,
and `perception_model_version` are optional diagnostics. Home Cortex can store
them. The client does not expect the server to branch on them.

V1 wraps the unchanged promotion payload in `hc.event` with event identity,
sequence and the current session fence, then posts to
`POST /client-interface/v1/messages`. Retry reuses the identical event. A new
session gets a new event while retaining the evidence ID and idempotency key.
The legacy `/v1/embodiments/{id}/session/evidence` route is used only in explicit
rollback mode.

## V1 provisioning

The deployed device origin is `https://home-cortex-0:8443`; enrollment uses
`https://home-cortex-0:8444/client-interface/v1/enroll`. The CA certificate must
arrive through a trusted operator channel. Hostname and CA validation are always
enabled. Redirects and plaintext V1 origins are rejected.

On the backend, `scripts/maintenance/client_interface.py prepare` creates TLS
material and the opt-in Docker overlay; `bootstrap` creates the first provisioner
for the existing body, with cortex-api stopped; `invite` uses the provisioner's
mTLS identity to create a one-use V1 invitation. See the backend
[deployment runbook](../home-cortex/docs/client-interface-deployment.md).
The invitation is a protected JSON file containing `invitation_id`, `token`,
`bootstrap_endpoint` and `embodiment_id`. Never paste its contents into logs.

```sh
home-cortex-client enroll \
  --invitation-file "$HOME/Library/Application Support/Home Cortex Client/bootstrap/invitation.json" \
  --ca-file "$HOME/Library/Application Support/Home Cortex Client/bootstrap/ca.crt" \
  --embodiment-id embodiment:macbook-0
home-cortex-client --source mac
```

The client generates its private key and PKCS#10 CSR locally with OpenSSL. A lost
enrollment reply retries with the same key/CSR. Successful enrollment writes:

```text
~/Library/Application Support/Home Cortex Client/credentials/
  identity.json       client_id, body, HTTPS origin, expiry and grants
  client.key          local private key
  client.csr          enrollment proof; reused on retry
  client.crt          issued client certificate
  ca.crt             original trusted CA
  protocol/
    receipts.sqlite3 durable command results and event identities
    process.lock     one client process per state directory
```

Directories require mode 0700; files require owner-only permissions (0600).
Symlink credential files are rejected. Override with `--credentials-dir` /
`HOME_CORTEX_CLIENT_CREDENTIALS_DIR` and `--state-dir` /
`HOME_CORTEX_CLIENT_STATE_DIR`. Receipts are bound to the client/body identity
and retained through the protocol's 24-hour retry window. Remove consumed
invitation files after enrollment. Certificates expire after 30 days. The backend
does not yet expose identity-preserving certificate rotation. A new invitation
and protected credential directory support operator re-provisioning/recovery,
followed by revocation of the old certificate; this creates a new principal and
must not be described as V1 certificate rotation.

The existing MacBook body now has the three implemented vision profiles
configured, and its replacement device credential grants still, clip and promotion
access. Enable the intended local profiles explicitly:

```sh
HOME_CORTEX_CLIENT_CAPABILITIES=vision.observe,vision.observe_clip \
  home-cortex-client --source mac
# Optional Stage 3, with the existing local analyzer and policy:
HOME_CORTEX_CLIENT_CAPABILITIES=vision.observe,vision.observe_clip,vision.autonomous_promotion \
HOME_CORTEX_CLIENT_ANALYZER=vision home-cortex-client --source mac
```

V1 stills are captured after request dispatch. Clips use actual contiguous JPEG
sample timestamps and the requested interval (the frozen protocol permits 1 ms
rounding). If no such interval exists, the client returns
`TEMPORARILY_UNAVAILABLE`; it never rewrites timestamps or silently shortens the
clip. Evidence ID, SHA-256, manifest and HCCLIP1 encoding remain unchanged.
Durable results suppress capture on redelivery and restart; an interrupted
receipt with an uncertain outcome fails safely without a second capture.

Rollback is explicit: set `HOME_CORTEX_CLIENT_INTERFACE=legacy`, the legacy HTTP
URL and household bearer key in the process environment. Never call this V1
conformance. `backend.py` and `commands.py` are isolated rollback adapters;
remove them and the flag after physical V1 acceptance. Local perception has
one shared implementation in both modes.
