# Local Agent Orchestration Implementation Plan

Status: implemented; scripted verification complete, real-Ollama smoke pending  
Target branch/worktree: `worktree-ollama-backend`  
Date: 2026-08-24

Path convention used below:

- Product `src/...` and `tests/...` paths are relative to
  `/Users/prithviseran/hunch-mcp/.claude/worktrees/ollama-backend`.
- `BENCH_ROOT` means `/Users/prithviseran/hunch-mcp/bench`, which is the active,
  lifecycle-hardened benchmark harness used for the current runs.
- Do not accidentally implement benchmark changes in the older
  `.claude/worktrees/ollama-backend/bench` copy. Synchronizing or replacing that
  copy is a separate, explicit integration decision.

## 1. Objective

Determine whether Qwen 3.5 9B becomes a viable Hunch controller when it acts as a
small coordinator and delegates execution to fresh, narrowly prompted instances
of the same model.

The experiment must answer three questions independently:

1. Does giving a fresh specialist only the relevant recipe and tools improve task
   completion over putting every recipe in the main agent's prompt?
2. Can Qwen 3.5 9B route tasks to the right specialist without losing the
   specialist's capability gain?
3. Does the architecture remain focus-free, bounded, observable, and safe enough
   to productize?

The first implementation is an Ollama experiment. It is not a new MCP tool or a
public SDK feature. Promotion into the public Hunch architecture happens only
after the benchmark gates in this document pass.

## 2. Preconditions

Before parallel implementation begins:

- Preserve and stabilize the existing uncommitted changes in:
  - `src/hunch/backends/ollama.py`
  - `tests/test_ollama_backend.py`
- Run the worktree's complete test suite and record the commit SHA used by every
  implementation agent.
- Freeze the recipe corpus and record its SHA-256. The static and orchestrated
  benchmark arms must use the same knowledge.
- Confirm all implementation agents use the active `BENCH_ROOT`; the worktree's
  older benchmark copy does not contain all runner lifecycle fixes from the last
  baseline.
- Do not modify the user's unrelated files or the main worktree's untracked site
  assets.
- Treat the current benchmark lifecycle, teardown, wake-lock, session-lock, and
  immutable-result behavior as required infrastructure. Do not replace it in an
  orchestration-specific runner.

The backend profile, context compaction, strict tool-channel, and no-progress
changes are dependencies of this experiment, not work to be reimplemented by an
orchestration agent.

## 3. Non-goals for the experiment

- Do not add `delegate` to `server.py`, `AGENT_TOOLS`, or `_DISPATCH`.
- Do not change the public count of 29 MCP tools.
- Do not expose orchestration through `Hunch.agent` yet.
- Do not run specialists in parallel.
- Do not add LangGraph, LiveKit, or another orchestration dependency.
- Do not create a second `Hunch` or `MacSession` per specialist.
- Do not fine-tune, swap LoRA adapters, or move from Ollama to llama.cpp/MLX in
  this phase.
- Do not train on or change benchmark task prompts/checkers.

## 4. Target architecture

```text
User task
   |
   v
Coordinator Ollama session
  system: compact routing and receipt contract
  tools:  delegate only
   |
   | DelegationRequest
   v
SpecialistOrchestrator
  - validates request and budgets
  - acquires one-Mac execution lease
  - creates a fresh child session
  - applies immutable tool scope
  - captures trace and disturbance delta
   |
   +-- applescript specialist -> applescript recipes + narrow native tools
   +-- native specialist      -> file/clipboard/direct OS tools
   +-- ax specialist          -> snapshot/find/act/app lifecycle
   +-- web specialist         -> CDP/credential/clipboard tools
   |
   v
_dispatch_core(shared_hunch, tool_name, arguments)
   |
   v
Bounded SpecialistReceipt -> coordinator
Full child trace          -> trace sink only
```

All logical agents use the same model tag and the same loaded Ollama weights, but
each has independent messages/KV context. All physical Mac actions continue to
flow through `_dispatch_core` on the original `Hunch` instance.

## 5. Architectural decisions

### 5.1 One internal delegation tool

The coordinator receives one internal tool named `delegate`. Its schema has a
specialist enum rather than exposing four unrelated tool implementations.

```json
{
  "specialist": "applescript | native | ax | web",
  "objective": "bounded description of the requested outcome",
  "context": {"allowlisted_key": "bounded value copied from the task"},
  "acceptance_criteria": ["observable condition"]
}
```

This tool exists only in the coordinator's local tool catalog. It must never enter
the public Hunch tool registry.

### 5.2 Instance-owned Ollama configuration

`HUNCH_OLLAMA_TOOL_PROFILE` is currently process-global and read at request time.
That is unsafe once multiple logical sessions use different scopes in one process.

Add immutable, instance-owned session configuration. Environment variables remain
backward-compatible defaults for the existing direct driver, but a configured
session never consults or mutates process-global profile state while running.

Minimum configuration:

```python
@dataclass(frozen=True)
class OllamaSessionConfig:
    role: str
    system_prompt: str
    tool_names: frozenset[str]
    allow_delegation: bool = False
```

Context window, history, result caps, strict mode, and model should also be
snapshotted into the instance/run configuration instead of being changed between
coordinator and child calls.

For reproducibility, snapshot the resolved model digest and Ollama server version,
not only the mutable model tag. Record two hashes: a `shared_config_sha256` over
the model digest, Ollama version, generation/sampler options, context limits,
strict mode, tool/result caps, operation limits, and recipe corpus (which must
match across arms), and an arm-specific `config_sha256` that additionally includes
the prompts and tool scopes intentionally changed by the experiment.

### 5.3 One shared Hunch, fresh child histories

Each delegation creates a fresh specialist backend/session but passes the original
`Hunch` instance. This preserves:

- the consent gate and host permission callback;
- simultaneous/focus-free policy;
- `MacSession` refs and disturbance accounting;
- CDP sessions and target binding;
- application and credential state.

Child message history must never be reused across delegations. Specialists should
re-snapshot rather than receiving AX refs from coordinator prose.

### 5.4 Whole-episode serialization

Acquire an execution lease around the entire specialist episode, not around each
individual tool call. Per-call locking would allow one specialist to invalidate
another specialist's window, CDP target, or AX refs between observation and action.

The experiment supports sequential delegation only:

- maximum active specialist episodes: 1;
- maximum nesting depth: 1;
- specialists cannot see or call `delegate`;
- the lease is released in `finally` after success, error, timeout, or interrupt,
  but only after any in-flight model request or physical tool call has terminated.

Timeout and interruption must not release the lease while execution is still
alive. A timed-out worker thread is not a terminated specialist: it may still be
inside an Ollama HTTP request or a physical tool call. Therefore:

- run the specialist synchronously while holding the lease;
- derive each Ollama request timeout from the episode's remaining deadline;
- check interruption/deadline before every model request and physical dispatch;
- after a timeout or interrupt, retain the lease until the in-flight request/tool
  call has actually returned;
- rely on the benchmark's existing process-group timeout for a truly wedged
  episode; never start a replacement specialist in the same process while the old
  one may still be running.

The experiment does not claim hard in-process cancellation. Hard cancellation is
a production blocker until the transport and every physical operation have a
termination mechanism that can prove the old episode is no longer executing.

### 5.5 Typed receipts, separate evidence levels

The coordinator receives a compact receipt, never the raw child conversation.

```python
class SpecialistStatus(StrEnum):
    REPORTED_COMPLETE = "reported_complete"
    VERIFIED_COMPLETE = "verified_complete"
    BLOCKED = "blocked"
    FAILED = "failed"
    TIMEOUT = "timeout"

@dataclass(frozen=True)
class SpecialistReceipt:
    delegation_id: str
    specialist: SpecialistKind
    status: SpecialistStatus
    summary: str
    evidence: tuple[str, ...]
    stop_reason: str
    turns: int
    tool_calls: int
    usage: Mapping[str, object]
    trace_id: str
```

`VERIFIED_COMPLETE` may be assigned only when trusted verification evidence is
available. The specialist saying "verified" is not trusted proof. In the initial
runtime, most successful receipts will be `REPORTED_COMPLETE`; the benchmark's
deterministic checker remains the authority for task success.

Receipt fields have explicit size limits. Secret-bearing tool results and raw
messages remain in the protected trace and are never copied into coordinator
context.

The specialist's final prose is untrusted and must not be copied verbatim into a
receipt. Receipt `summary` and `evidence` are constructed by the orchestrator from
an allowlist of structured event metadata (tool name, success/error class,
redacted target identity, checker/verifier result, counts, and stop reason).
Clipboard contents, credential values, AppleScript output, DOM/AX text, paths not
explicitly supplied by the user, screenshots, and arbitrary tool-result text are
excluded. Redaction is defense in depth, not the basis of the confidentiality
claim.

`DelegationRequest.context` is likewise an allowlisted structured mapping, not an
arbitrary bag of model-generated strings. The initial experiment permits only
keys defined by the selected specialist contract and rejects unknown keys. Values
must be bounded scalars copied from the original task or trusted runtime metadata;
the coordinator cannot assert that an arbitrary value is "non-secret."

### 5.6 Deterministic actions before agent delegation

The specialist registry may later point a known operation to deterministic code
instead of another model. That is not required for the first experiment, but the
contracts must not assume every specialist is model-backed.

Long-term routing order:

```text
known deterministic operation -> execute directly
known adaptive domain          -> run constrained specialist
unknown domain                 -> coordinator reports blocked
```

## 6. Specialist registry and scopes

Implement an immutable registry. Every tool name must be validated against
`_DISPATCH` at startup/test time.

| Specialist | Initial tools | Prompt/recipe input |
|---|---|---|
| `applescript` | `applescript`, `clipboard_get`, `clipboard_set`, `list_apps`, `notify_user` | Verified Notes, Reminders, Calendar, Mail, and shell/path recipes |
| `native` | `file_op`, `trash`, `open_file`, `reveal_in_finder`, `clipboard_get`, `clipboard_set`, `list_apps`, `launch_app`, `quit_app`, `notify_user`, plus `applescript` restricted by prompt to direct file reads/`do shell script` | Direct-operation priority, literal-path, shell quoting, and collision rules |
| `ax` | `snapshot`, `find`, `act`, `list_apps`, `launch_app`, `quit_app`, `notify_user` | Observe-act-verify, label/control grounding, stale-ref recovery |
| `web` | Existing restricted web/CDP profile, credential helpers, clipboard, notification | Re-snapshot after navigation, combobox/form and clipboard patterns |

Do not give specialists `simultaneous_mode`. Do not include `request_focus` in
the focus-free experimental scopes. The host owns these decisions.

Store prompts and recipes as package resources, not hard-coded string fragments
scattered through the orchestrator. Preserve a single canonical recipe corpus so
the static and split variants can be generated and hashed from the same sources.

## 7. Execution lifecycle and budgets

One coordinator run follows this lifecycle:

1. Snapshot the run configuration, original task, deadline, model, and context
   limits.
2. Start a coordinator session with only the compact routing prompt and `delegate`.
3. Validate each `DelegationRequest` before executing it.
4. Reject recursion, unknown roles, oversized context, secrets, exhausted budgets,
   or disallowed tools without touching the Mac.
5. Acquire the one-Mac execution lease.
6. Record disturbance counters and create a fresh specialist session.
7. Give the specialist the original user task, bounded delegated objective,
   acceptance criteria, its focused prompt, and only its tool schemas.
8. Run all real tools through scope, deadline, budget, and host-permission checks,
   reserve their physical-operation cost, and only then call `_dispatch_core` on
   the original Hunch instance.
9. Capture the full child trace under a child `trace_id`.
10. Build a bounded receipt from `AgentResult`, actual events, usage, and disturbance
    deltas.
11. After any in-flight request/tool call has terminated, release the lease in
    `finally` and return only the receipt to the coordinator.
12. Finish when the coordinator returns, or when the shared deadline/delegation/
    turn budget is exhausted.

Initial configurable limits:

- nesting depth: 1;
- concurrent specialist episodes: 1;
- delegations per task: 3;
- automatic retry: at most 1, only for explicitly retryable transport failures;
- never retry consent refusal, unsafe-action refusal, invalid request, or ordinary
  task failure;
- model-turn budget: 40 per execution agent, matching the direct control's
  existing `max_turns` meaning;
- physical-operation budget: configured separately and enforced across all
  specialists in one coordinator run;
- outer benchmark timeout: unchanged at 900 seconds;
- child deadlines must leave time for the coordinator to receive the receipt and
  exit.

`interrupt()` must propagate to the coordinator and the active specialist. The
current Ollama HTTP call can block until its request timeout and is only interruptible
between turns. The request timeout must be capped to the remaining episode deadline,
but the lease remains held until the request returns. Record the lack of hard
cancellation as a known production blocker rather than hiding it.

Physical-operation accounting is defined before implementation:

- one ordinary `_dispatch_core` call costs one operation;
- `act(actions=[...])` costs the number of action items, with a minimum cost of one;
- `file_op(batch=[...])` and multi-path `trash`/`reveal_in_finder` cost the number
  of requested items, with a minimum cost of one;
- malformed, unknown, or out-of-scope calls rejected before dispatch cost zero
  physical operations but remain visible in error telemetry;
- retries do not refund operations that already reached `_dispatch_core`;
- reserve the full calculated cost atomically before dispatch and reject the
  entire call if insufficient budget remains.

All three benchmark arms retain the same model-turn limit and physical-operation
limit. Coordinator-only routing turns are measured separately and are not falsely
reported as physical actions.

## 8. Work packages for implementation agents

Agents must not edit files outside their assigned ownership without coordinating
with the integration owner. Each agent must state the base commit, preserve other
agents' work, and add tests with its change.

### WP0 — Stabilize the current Ollama baseline

Owner: one integration/preparation agent  
Dependencies: none  
Exclusive files:

- `src/hunch/backends/ollama.py`
- `tests/test_ollama_backend.py`

Deliverables:

- Finish and test the existing profile, compaction, strict-channel,
  no-progress, and lifecycle-related diff.
- Run the entire test suite.
- Record the clean base commit for WP1-WP5.
- Make no orchestration changes in this package.

Exit gate: worktree tests pass and implementation agents share one immutable base
SHA.

### WP1 — Instance-scoped Ollama runtime

Owner: backend agent  
Dependencies: WP0  
Exclusive files:

- `src/hunch/backends/ollama.py`
- optional new `src/hunch/backends/ollama_loop.py`
- `tests/test_ollama_backend.py`
- optional new `tests/test_ollama_loop.py`

Deliverables:

- Introduce immutable per-session system prompt and tool catalog/scope.
- Preserve environment-driven defaults for normal direct Ollama use.
- Support an injected/internal tool dispatcher so the coordinator can expose
  `delegate` without adding it to Hunch tools.
- Ensure two sessions can have independent prompts, tools, histories, and usage.
- Preserve `_dispatch_core` as the only real Hunch execution engine.
- Enforce the existing `can_use_tool` permission callback before every physical
  tool dispatch in the experiment. The scoped dispatcher must apply checks in
  this order: schema/request validation, immutable specialist scope, deadline and
  budget-availability check, host permission callback, atomic budget reservation,
  then `_dispatch_core`. Permission denial consumes no physical-operation budget.
- Define a small adapter for the callback's existing Allow/Deny protocol so the
  synchronous Ollama loop does not silently bypass an async host callback. A
  denied or failed permission check never reaches `_dispatch_core` and is never
  automatically retried.
- Keep current strict parsing, context compaction, no-progress detection, image
  handling, and telemetry behavior unchanged.

Exit tests:

- instance scope overrides environment default;
- two instances cannot observe one another's profile or history;
- out-of-scope calls never reach `_dispatch_core`;
- every allowed physical call reaches `can_use_tool` exactly once before
  `_dispatch_core`, and denial prevents dispatch;
- direct Ollama behavior remains backward compatible;
- current backend tests still pass.

### WP2 — Contracts, specialist registry, and recipe packaging

Owner: contracts/recipe agent  
Dependencies: WP0; may run in parallel with WP1  
Exclusive files:

- new `src/hunch/orchestration/__init__.py`
- new `src/hunch/orchestration/contracts.py`
- new `src/hunch/orchestration/specialists.py`
- new `src/hunch/orchestration/recipes/*`
- new `tests/test_orchestration_contracts.py`
- `BENCH_ROOT/recipes/*` only after coordinating canonical-source ownership

Deliverables:

- Implement immutable kinds, specs, requests, receipts, and statuses.
- Enforce objective, context, acceptance-criteria, receipt, and evidence bounds.
- Implement the context-key allowlists and structured, redacted receipt builder;
  do not derive receipt evidence from arbitrary child prose or raw tool results.
- Implement deterministic serialization suitable for transcript storage.
- Build the four specialist specs and validate every physical tool against
  `_DISPATCH`.
- Split/generate specialist recipe views from one frozen canonical corpus.
- Add recipe SHA-256 generation and verification.
- Distinguish reported completion from trusted verification.

Exit tests:

- invalid enum/status/request rejected;
- specs are immutable;
- serialization is stable;
- receipt never contains raw child messages;
- receipt/context serialization rejects secret-bearing and unknown fields;
- `VERIFIED_COMPLETE` requires trusted evidence;
- every specialist tool is a real Hunch dispatch tool;
- `delegate` is absent from MCP and `_DISPATCH`.

### WP3 — Specialist orchestrator and execution lease

Owner: orchestration agent  
Dependencies: WP1 and WP2  
Exclusive files:

- new `src/hunch/orchestration/execution.py`
- new `src/hunch/orchestration/orchestrator.py`
- new `tests/test_orchestration.py`

Deliverables:

- Implement scoped dispatch into `_dispatch_core` using the same Hunch object.
- Implement the whole-episode execution lease with `finally` release.
- Construct a fresh child session for every delegation.
- Enforce depth, delegation, action, turn, retry, and deadline budgets.
- Keep the execution lease until an in-flight request/tool call truly returns;
  timeout or interrupt must never merely abandon a live worker and release it.
- Propagate interrupts to the active child.
- Capture child traces separately and return bounded typed receipts.
- Record disturbance deltas without allowing the specialist to modify policy.
- Use dependency-injected backend/session factories so unit tests need no Ollama
  daemon or UI.

Exit tests:

- allowed call reaches `_dispatch_core` once with the same Hunch instance;
- AppleScript child cannot call AX/web tools;
- consecutive delegations have fresh histories;
- full trace goes only to the trace sink;
- retry occurs once for a retryable transport error and never for refusal/task
  failure/invalid request;
- lock releases after error and interruption;
- a simulated timed-out but still-running request keeps the lock and prevents a
  replacement episode until the request terminates;
- two threaded fake episodes prove maximum Mac concurrency is one;
- no specialist can delegate recursively.

### WP4 — Coordinator session and scripted integration

Owner: coordinator agent  
Dependencies: WP1-WP3  
Exclusive files:

- new `src/hunch/orchestration/coordinator.py`
- new `tests/test_orchestration_integration.py`

Deliverables:

- Implement the compact coordinator prompt and internal `delegate` schema.
- Give the coordinator no Hunch tools and no full Hunch playbook.
- Pass the original task through to every child alongside the delegated objective.
- Deliver receipts as tool results without child transcript leakage.
- Aggregate coordinator and child usage without confusing cumulative tokens with
  peak request context.
- Emit stable coordinator/child invocation IDs and event metadata.

Scripted fake-Ollama acceptance path:

1. Coordinator calls `delegate(applescript, ...)`.
2. Specialist calls `applescript` through its native tool channel.
3. Fake `_dispatch_core` succeeds.
4. Specialist performs/readies verification evidence and exits.
5. Coordinator receives only the typed receipt and finishes.

Assert that child trace content is present in the trace sink but absent from the
coordinator's prompt/messages.

### WP5 — Benchmark adapter, paired runner, and analysis

Owner: benchmark agent  
Dependencies: WP2 contracts may start first; final integration depends on WP4  
Exclusive files:

- `BENCH_ROOT/local_driver.py` or a thin shared driver module
- new `BENCH_ROOT/orchestrated_driver.py` if separation is cleaner
- new `BENCH_ROOT/adapters/hunch-ollama-orchestrated.json`
- `BENCH_ROOT/run.py` only for backward-compatible result fields/events
- new `BENCH_ROOT/run_orchestration_experiment.py`
- new `BENCH_ROOT/analyze_orchestration.py`
- relevant benchmark unit tests

Deliverables:

- Keep `BENCH_ROOT/run.py::run_task` as the owner of setup, checker, and teardown.
- Keep `WakeLock`, session-lock detection, lifecycle preflight, immutable result
  streams, fatal abort rules, and process-group timeout behavior.
- Add two orchestrated route modes:
  - `oracle`: task metadata selects the specialist directly, bypassing coordinator;
  - `coordinator`: raw task text goes to the coordinator, which selects a role.
- In coordinator mode, do not reveal the benchmark's `_tool_profile` result to the
  coordinator.
- Add paired arm scheduling with deterministic alternating order per task/trial.
- Route the `static_recipes` arm through the same physical-operation accounting
  wrapper (without delegation or specialist scoping) so its budget has identical
  units and rejection behavior.
- Make resume identity include arm plus recipe/shared-configuration hashes so rows
  cannot collide; the shared hash includes the clean source commit and active
  benchmark-code hash.
- Record the resolved Ollama model digest, Ollama server version, generation
  options, macOS/hardware identity, and physical-operation budget in every row and
  include the stable execution configuration in the configuration hash.
- Preserve existing top-level fields and add orchestration metadata.
- Store full coordinator and specialist conversations in transcripts, tagged with
  parent/child IDs. Continue emitting physical tool events for existing auditing.
- Add paired outcome analysis, net wins, McNemar exact test, and task-bootstrap
  confidence intervals.

Required result additions:

```json
{
  "arm": "static_recipes | oracle_specialist | orchestrated",
  "experiment_id": "...",
  "recipe_sha256": "...",
  "shared_config_sha256": "...",
  "config_sha256": "...",
  "model_tag": "qwen3.5:9b",
  "model_digest": "...",
  "ollama_version": "...",
  "orchestrator_version": "...",
  "orchestration": {
    "route": "...",
    "route_correct": true,
    "delegations": 1,
    "retries": 0,
    "coordinator_turns": 2,
    "specialist_turns": 8,
    "specialist_tool_calls": 7,
    "physical_operations": 7,
    "physical_operation_limit": 40,
    "verified_receipts": 0,
    "receipt_failures": 0,
    "max_depth": 1,
    "serial_execution": true
  },
  "usage": {
    "coordinator": {},
    "specialists": [],
    "total_input_tokens": 0,
    "total_output_tokens": 0,
    "peak_context_chars": 0
  }
}
```

Token work is summed; peak context is the maximum single request, never the sum of
independent contexts.

### WP6 — Integration and safety review

Owner: integration/review agent  
Dependencies: WP1-WP5  
Files: cross-cutting changes only after work-package owners finish

Deliverables:

- Rebase/integrate work packages without overwriting the WP0 backend fixes.
- Run targeted tests after each integration, then the complete suite.
- Verify public MCP tools remain exactly 29 and equal `_DISPATCH`.
- Verify `delegate` exists only in the private coordinator catalog.
- Verify no second Hunch/MacSession is constructed.
- Verify the original gate, host permission callback, simultaneous mode, CDP
  session, and disturbance counter are reused.
- Verify coordinator and specialists cannot expose credential values in receipts.
- Run static and lifecycle preflight before every real-Mac smoke/benchmark.
- Review every new timeout/interrupt/lock `finally` path.

## 9. Safe parallel execution order

```text
WP0 stabilize baseline
  |
  +-------------------+
  v                   v
WP1 Ollama runtime   WP2 contracts/recipes
  |                   |
  +---------+---------+
            v
       WP3 orchestrator
            |
            v
       WP4 coordinator
            |
            +------------------+
            v                  v
       WP5 benchmark      scripted integration
            +------------------+
                       v
               WP6 integration/review
                       |
                       v
                 paired experiment
```

WP5 may build adapter/result scaffolding while WP3-WP4 are underway, but it must
not invent duplicate orchestration contracts.

## 10. Test matrix

### Unit and invariant tests

- Existing full test suite remains green.
- Exact MCP tool count remains 29.
- MCP tool names still equal `_DISPATCH` names.
- `delegate` is absent from both.
- Scope validation happens before physical dispatch.
- Environment configuration remains backward compatible.
- Context/history/usage are isolated per logical session.
- Receipt size and evidence bounds are enforced.
- No raw child transcript enters coordinator history.
- Fresh child session per delegation.
- One active Mac episode maximum.
- Shared Hunch/gate/policy/session identity.
- Interrupt, deadline, exception, retry, and lock-release paths.
- Focus-stealing action remains refused in simultaneous mode.
- Disturbance delta remains zero in the focus-free fake/integration path.

### Real-Mac smoke

Run one lifecycle-preflighted task per specialist before a benchmark:

- AppleScript: one verified Notes/Calendar operation.
- Native: one filesystem or clipboard operation.
- AX: one Settings operation using snapshot/find/act.
- Web: one local CDP fixture operation.

For every smoke task require:

- deterministic checker success;
- teardown success;
- zero cursor moves;
- no unexpected focus switch;
- one specialist active at a time;
- full coordinator/child transcript present;
- coordinator context contains receipt only.

## 11. Benchmark experiment

### 11.1 Arms

Use the same Qwen tag and digest, Ollama options, strict tool mode, task prompts,
checkers, outer timeout, model-turn limit, physical-operation budget, and recipe
corpus in every arm.

"Same Qwen tag" is insufficient on its own because a tag can be moved. Abort a
paired experiment if the resolved model digest or Ollama version/configuration
hash differs across its arms.

1. `static_recipes`: current direct agent with the full frozen recipe corpus as
   `--exemplars`/system suffix.
2. `oracle_specialist`: benchmark metadata directly chooses the specialist. This
   measures the execution architecture without routing errors.
3. `orchestrated`: compact coordinator chooses the specialist and receives a
   bounded receipt.

The plain 30/75 baseline remains historical context, but the paired experiment's
static arm must be rerun under the same host and corpus conditions.

### 11.2 Pilot

Start with the 12 native/AppleScript-profile tasks:

`01, 02, 03, 04, 12, 13, 15, 16, 17, 18, 19, 20`

Run three trials per task per arm. Pair by task/trial and alternate arm order
deterministically to reduce cache, thermal, and app-state ordering effects.

Do not cherry-pick only prior failures. Easy ceiling tasks are needed to detect
regressions.

### 11.3 Pilot gates

Safety gates are absolute:

- zero teardown failures;
- zero cursor movement;
- zero un-restored unsafe state changes;
- no concurrent specialist episodes;
- no public tool-surface changes.

Capability/reliability gates:

- `oracle_specialist` achieves at least five net paired wins over
  `static_recipes` across the 36 paired native rows (about +14 percentage points);
- only after that oracle gate passes, `orchestrated` retains at least 80% of the
  oracle arm's net capability gain, where retention is
  `(orchestrated_pass_rate - static_pass_rate) /
  (oracle_pass_rate - static_pass_rate)` on the same paired rows;
- orchestration does not increase agent-error rate;
- route accuracy is reported separately from specialist completion;
- recipe and shared-configuration hashes match across arms; arm-specific
  configuration hashes are recorded but are expected to differ.

If the oracle denominator is zero or negative, retention is undefined and the
orchestrated arm cannot pass promotion. Report the value as `null`, not zero or an
infinite percentage. The five-net-win threshold is an engineering pilot gate, not
a claim of statistical significance; McNemar and task-bootstrap results must be
reported alongside it, with repeated trials clustered by task for uncertainty
analysis.

Context/efficiency reporting, not an initial hard rejection gate:

- coordinator peak request context;
- specialist peak request context;
- total input/output tokens across both agents;
- wall time and time to first physical action;
- coordinator turns, specialist turns, delegations, retries, and compactions.

### 11.4 Expansion

If the pilot passes, run all 25 default-runnable tasks, three trials per arm.

- Web is a ceiling/control domain because the direct baseline was already strong.
- AX measures whether focused grounding instructions reduce loops and wrong-target
  behavior.
- Analyze paired per-task outcomes; do not treat all repeated rows as independent.

Use net paired wins, per-task pass rates, McNemar exact tests, and bootstrap
confidence intervals over tasks.

## 12. Promotion criteria

Only begin public SDK architecture work when:

- the pilot and expanded suite pass their safety gates;
- orchestration has a repeatable capability gain over static recipes;
- coordinator routing preserves most of the oracle specialist gain;
- traces show the gain comes from better execution rather than checker leakage or
  benchmark metadata;
- context isolation is real (no child transcript copied into coordinator history);
- timeouts and interruption are acceptable or have an explicit production fix;
- the permission callback is enforced for every physical specialist tool call.

If oracle specialists improve but coordinator routing does not, train or redesign
the coordinator separately. If neither oracle specialists nor static recipes move
the 9B model materially above baseline, test the same frozen architecture and
prompts with the 27B model before investing in fine-tuning.

## 13. Definition of done for the experiment

- WP0-WP6 deliverables and tests are complete.
- Full test suite passes from a clean checkout.
- The direct Ollama adapter still behaves as before.
- The orchestrated adapter can be started by one background shell command and
  inherits the proven wake-lock/lifecycle behavior.
- Every run is reproducible from recorded model, recipe hash, configuration hash,
  commit SHA, task set, arm, and trial.
- Every delegation has a parent ID, child trace ID, typed receipt, scoped tools,
  usage, deadline, and disturbance delta.
- No specialist runs concurrently or recursively.
- The paired pilot report clearly distinguishes routing quality, specialist
  execution quality, safety, context occupancy, total token work, and latency.
- A promotion/no-promotion decision can be made without reading raw traces.

## 14. Agent handoff checklist

Every implementation agent should include this in its handoff:

- base commit SHA and work package number;
- files owned and files changed;
- contract assumptions used;
- tests added and exact commands run;
- known limitations or follow-up work;
- whether any public API/tool/schema changed;
- whether any real Mac action was run;
- confirmation that unrelated work was preserved;
- commit SHA containing the completed work package.
