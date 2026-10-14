# Healthcare outbound voice agent (LiveKit + Opik)

An AI voice agent that phones a patient, confirms who it is speaking to, explains their recent
blood glucose and HbA1c results in plain language, and books a doctor's consultation through a
(simulated) scheduling tool. After the call, the call is analyzed (was an appointment booked?)
and logged to Opik as a trace, where online evaluation rules score it.

Built for the "Outbound Voice Agent Assignment"
([requirements and traceability](docs/assignment_requirements.md)).

> **Verification status (2026-10-12).**
>
> **Verified against the real services:**
>
> - Opik traces, plus the two online rules executed by Opik with scores attached (a false
>   booking claim scores 0).
> - OpenAI LLM agent, post-call analysis, and TTS/STT.
> - LiveKit authentication, worker registration and dispatch.
>
> **Not verified yet:** a real phone call, and with it real call recording. The configured
> outbound SIP trunk has no real provider behind it.
>
> Local suite: 249 passed. Real external tests: 4 passed. Details are in the
> [final verification report](docs/final_end_to_end_verification_report.md).

## How it works

```
place_call.py ─► call_manager: validate patient, allowlist, explicit agent dispatch
                      │  (call_id + patient in authenticated job metadata)
                      ▼
agent worker (app/agent.py) ─► CreateSIPParticipant ─► SIP trunk ─► patient's phone
   │  AgentSession: OpenAI STT ─► LLM (+ tools) ─► TTS, Silero VAD, audio recording
   │  tools ─► CallTools rules ─► scheduling backend (SQLite simulator)
   ▼  on_session_end
post-call pipeline: transcript + recording ─► analysis ─► call_report.json ─► Opik trace
                                                                     └► Opik online rules
```

- **LiveKit Agents** runs the realtime conversation: audio streaming, turn-taking,
  interruptions and the tool-calling loop. **SIP** is how phone calls are set up. An **outbound
  SIP trunk** (from Twilio, Telnyx and similar providers) connects LiveKit to the phone network.
  The worker asks LiveKit to dial the patient. When they answer, the phone becomes a participant
  in the agent's room.
- **STT** (`gpt-4o-mini-transcribe`) turns the patient's speech into text. The **LLM**
  (`gpt-4.1-mini`) decides what to say and which tool to call. **TTS** (`gpt-4o-mini-tts`)
  speaks the reply. A text pipeline was chosen over a speech-to-speech model so every step is
  inspectable and testable ([why](docs/architecture.md#key-decisions)).
- **Tools**: `verify_identity`, `check_availability`, `book_appointment`,
  `record_patient_decision`, `end_call`. The rules are enforced in Python, not just in the
  prompt. Health results are only given to the model *after* identity is confirmed. Booking
  requires an offered slot and explicit patient confirmation. A booking exists only if the
  backend commits it (idempotent, one per slot).
- **Post-call analysis** derives the outcome from recorded facts (dial result, tool events,
  backend record). It never infers booking from the transcript. An optional LLM summary adds
  qualitative notes. Running it twice gives the same result.
- **Opik**: [`app/observability/opik_integration.py`](app/observability/opik_integration.py) is
  the only module that touches the tracing SDK. Its interface is `log_call(record, analysis)`.
  Each call becomes one trace: metadata and patient variables, transcript, recording reference
  plus audio attachment, `tool:*` spans with results, and the post-call analysis.
- **Online evaluation**: `booking_claim_accuracy` is an Opik Python-code rule that checks
  whether what the agent *said* about booking matches what the backend *did*.
  `medical_advice_safety` flags diagnoses or medication advice. An LLM-as-a-judge rule is
  optional. See [docs/opik_evaluation.md](docs/opik_evaluation.md).
- **Why no LangGraph?** One agent, five tools, a linear flow. LiveKit already runs the loop, and
  the critical rules live in plain Python. More in
  [docs/interview_guide.md](docs/interview_guide.md#c-agent-architecture).

## Requirements

- Python 3.10–3.13 (developed on 3.12)
- For real calls: a [LiveKit Cloud](https://cloud.livekit.io) project with SIP, an outbound SIP
  trunk from a telephony provider with an allowed caller ID, and a phone number you are
  authorized to call
- An OpenAI API key (voice pipeline and post-call summary)
- An [Opik](https://www.comet.com/opik) workspace and API key (or self-hosted Opik)

None of these are needed for the offline demo and tests.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env        # then fill in what you have
```

Configuration reference: [`.env.example`](.env.example). Each command checks only what it
needs and names any missing variables. `.env` is git-ignored.

## Run the offline demo (simulation, no phone call)

```bash
python scripts/run_demo.py                       # booked, callback, booking failure, false claim
python scripts/run_demo.py --scenario all        # all 10 scenarios
python scripts/run_demo.py --list
python scripts/run_demo.py --llm openai          # the real model drives the agent's side
```

This runs the real agent, tools and scheduler inside a text-only LiveKit `AgentSession`. Each
scenario prints the patient input, transcript, tool calls with backend results, post-call
analysis and Opik status. It writes evidence to `artifacts/demo/` and full reports to
`artifacts/calls/<call_id>/`. Simulated calls are tagged `mode:simulation` everywhere. The
default agent side is a scripted stand-in for the LLM. With Opik configured, the same command
sends traces to Opik.

## Place a real call

Only call numbers you own or are authorized to call. The destination must be listed in
`AUTHORIZED_TEST_NUMBERS`, or nothing is dialed.

```bash
# terminal 1: the agent worker
python -m app.agent dev

# terminal 2: dispatch the agent for the synthetic patient, pointed at your test phone
python scripts/place_call.py --patient data/demo_patient.json --phone +15551234567
```

`place_call.py` waits for the call to finish and prints the outcome, booking, recording status
and Opik trace ID. The report and `recording.ogg` are in `artifacts/calls/<call_id>/`. Use
`SCHEDULER_SIMULATION=booking_failure` or `no_slots` (restart the worker) to demo failure paths
on a real call. Patient input format: [`data/demo_patient.json`](data/demo_patient.json)
(synthetic).

## Opik

```bash
python scripts/setup_opik_rules.py                 # install the online evaluation rules (once)
python scripts/run_demo.py --scenario appointment_booked false_booking_claim
python scripts/verify_opik.py --call-id <call_id> --wait 180
python scripts/replay_telemetry.py --all-unsent    # re-send reports after an Opik outage
```

In the Opik UI: project `healthcare-outbound-calls` → Traces → `outbound_call`. Tool spans sit
under `conversation`. The online-rule scores (`booking_claim_accuracy`,
`medical_advice_safety`) appear as feedback scores on the trace. A correct agent scores 1; the
deliberately faulty `false_booking_claim` scenario should score 0 on booking claims.

## Tests

```bash
pytest                                       # 249 offline tests
RUN_EXTERNAL_TESTS=1 pytest -m external -v   # real Opik / OpenAI / LiveKit (no calls placed)
python scripts/verify_openai_conversation.py # real model vs an LLM-played patient (text only)
python scripts/make_test_report.py           # writes artifacts/test_report.json
ruff check . && ruff format --check .
```

Results, the scenario-to-test map and manual verification checklists are in
[docs/testing.md](docs/testing.md).

## Project layout

```
app/
  agent.py              LiveKit worker: dial, session, tool wrappers, session-end hook
  call_manager.py       validation, allowlist, explicit agent dispatch
  tools.py              tool rules + ToolEvent recording
  scheduling.py         simulated scheduler (SQLite, idempotent, exclusive slots)
  patient.py            patient model, biomarker wording
  prompts.py            agent instructions (no health data)
  call_state.py         authoritative per-call facts
  transcript.py, recording.py
  post_call.py          outcome + analysis
  pipeline.py           idempotent finalize: analyze → save → Opik
  simulation.py, scenarios.py   text-mode runs with a scripted LLM
  observability/
    opik_integration.py         the Opik module
    evaluation.py               online rule definitions + installer
    online_metrics/*.py         code uploaded to Opik as rules
scripts/                run_demo, place_call, setup_opik_rules, verify_opik, replay_telemetry, make_test_report
tests/                  unit / integration / e2e / external
docs/                   architecture, requirements, testing, Opik evaluation, interview guide, troubleshooting
```

## Security and privacy

- Secrets come only from the environment. They are masked in errors and logs, and `.env` is
  git-ignored.
- An allowlist of destination numbers is enforced in the call manager and again in the worker.
- Room names, trace names and logs never contain phone numbers, names or health values. Reports
  and traces carry a pseudonymous `patient_id` and a masked number.
- Demo data is synthetic. Real use needs recording consent, retention and deletion policies,
  access-controlled audio storage, and vendor agreements for PHI
  ([details](docs/architecture.md#data-handling)).
- The agent is an appointment assistant: it does not diagnose or advise on medication, and it
  directs emergencies to `EMERGENCY_NUMBER`.

## Known limitations

- A real phone call, and with it recording, has not been verified (see above). Opik
  rules and OpenAI are verified against the real services.
- On macOS with python.org Python, run `Install Certificates.command` or set
  `SSL_CERT_FILE=$(python -m certifi)`. Otherwise LiveKit connections fail TLS
  ([troubleshooting](docs/troubleshooting.md)).
- Booking-claim and safety rules are pattern-based. They're explainable and tested, but not
  exhaustive.
- Identity is confirmed by name only. Production should add a second factor.
- Recordings are stored on local disk. `place_call.py --wait` assumes the worker is on the same
  machine.

More: [troubleshooting](docs/troubleshooting.md) · [implementation plan](docs/implementation_plan.md)
· [interview guide](docs/interview_guide.md)
