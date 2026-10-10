# Qwen Distillation Data Pilot

## Objective

Create a quarantined 25-task data pilot that:

- is intentionally difficult for the current small Qwen model;
- is reliably solvable by a substantially larger open-weight teacher;
- uses the real Hunch system prompt, tool schemas, and execution loop;
- has deterministic setup, external verification, and teardown;
- remains separate from the canonical benchmark suite; and
- can produce execution-validated supervised fine-tuning records.

This pilot is a go/no-go experiment. It should prove that a larger open-weight
model can generate consistently correct Hunch trajectories before the dataset
is expanded or Qwen is fine-tuned.

## Teacher model

Use **Kimi K3** as the primary teacher candidate.

Kimi K3 is preferable to a closed model because its full weights are published
under the Kimi K3 License. The license grants broad rights to use, modify,
fine-tune, and create derivative works from the software and model weights. It
does not contain an explicit anti-distillation clause. It does contain
commercial conditions for large model-service businesses and very large
products, so the exact license revision used by the experiment must be pinned
and reviewed before production use.

References:

- [Kimi K3 license](https://huggingface.co/moonshotai/Kimi-K3/blob/main/LICENSE)
- [Kimi K3 model card](https://huggingface.co/moonshotai/Kimi-K3)
- [Kimi K3 tool-calling guide](https://platform.kimi.ai/docs/guide/kimi-k3-quickstart)

### Inference route

Kimi K3 has 2.8 trillion total parameters and activates approximately 104
billion parameters per token. It is not practical to run on the development
Mac. Use one of these routes:

1. A temporary multi-GPU cloud deployment using the published weights.
2. A hosted Kimi K3 provider whose contract explicitly permits using model
   outputs to train or distill another model.

Self-hosting the published weights provides the cleanest licensing posture. If
a hosted API is used, its terms apply independently of the model-weight
license. Obtain written confirmation that generated outputs may be used to
fine-tune Qwen before collecting training data.

Pin the following information in the experiment metadata:

- model repository and exact revision;
- license revision or file hash;
- tokenizer revision;
- inference engine and version;
- quantization, if any;
- reasoning-effort setting;
- provider and provider terms, if hosted; and
- sampling configuration.

## Phase 0: prerequisites

### Correct the evaluation harness

Fix the Ollama AppleScript cutoff classifier before measuring the small-model
baseline. Runtime or access error `-1723` must not advance the compile-error
counter. Only genuine syntax or compile failures such as `-2741` should count
toward the compile-loop cutoff.

### Seal the canonical evaluation suite

Treat `bench/tasks` as evaluation-only:

- never read it while generating the new tasks;
- do not reuse its titles, tokens, filenames, or prompt wording;
- do not create trivial paraphrases of its prompts;
- do not train on canonical transcripts or checker output; and
- continue using the canonical suite only for post-training evaluation.

The benchmark runner already supports an alternate task root through
`--tasks-dir`. The data pilot must always use that option.

## Phase 1: five-task Kimi integration gate

Do not implement or run all 25 tasks until Kimi can complete the real Hunch
tool loop correctly.

### Build a minimal teacher driver

The driver should:

1. Send the real Hunch system prompt and native tool schemas to Kimi K3.
2. Dispatch returned tool calls through Hunch's shared execution engine.
3. Append complete tool results to the conversation.
4. Continue until Kimi produces a final response or reaches a bounded cutoff.
5. Emit the same stream-JSON shape used by `bench/run.py` where practical.
6. Record the complete observable conversation and usage metadata.

Kimi K3 requires the complete assistant message to be passed back during a
multi-turn tool loop, including `reasoning_content` and `tool_calls`. Preserve
that message while executing the episode. Do not include `reasoning_content`
in the final Qwen training examples.

### Initial five tasks

Create one representative task from each family:

1. Exact stateful Apple Notes edit.
2. Reminders reconciliation from a file manifest.
3. Calendar event creation from structured input.
4. Safe Mail draft creation from multiple sources.
5. Cross-app composition producing two independently verified outputs.

### Integration acceptance criteria

Proceed to the full pilot only if:

- every reference solution passes its checker;
- Kimi passes at least four of the five tasks;
- native tool calls and arguments are captured without conversion loss;
- no email is sent;
- there are zero focus switches and cursor movements;
- teardown removes all task-created state; and
- the retained traces contain concise, reusable actions rather than accidental
  success through retries or unrelated state.

## Phase 2: repository structure

Keep versioned definitions separate from generated runtime state:

```text
research/distill/
├── README.md
├── pilot_tasks.json
├── materialize.py
├── hook.py
└── extract_training_data.py

bench/distill/tasks/       # generated; passed through --tasks-dir
tests/test_distill_pilot.py
```

`pilot_tasks.json` is the source of truth. `materialize.py` should produce
ordinary benchmark task directories:

```text
task.json
prompt.txt
setup.sh
check.sh
teardown.sh
```

The shell files should be thin wrappers around the shared `hook.py`. Avoid
copying large AppleScript setup and verification programs into every task.

## Phase 3: task schema

Each task definition should contain:

```json
{
  "id": "d01",
  "family": "notes",
  "description": "Short human-readable description",
  "difficulty_tags": [
    "stateful-edit",
    "exact-output",
    "cross-source"
  ],
  "prompt": "The exact prompt shown to the model",
  "timeout_s": 300,
  "files": {},
  "initial_state": {},
  "expected_state": {},
  "forbidden_state": {}
}
```

Initially support only these resource types:

- files and directories;
- Apple Notes;
- Reminders lists and reminders;
- Calendar events; and
- Mail drafts.

Use fixed, deterministic task values for the 25-task pilot. Parameterized
random generation can be added when expanding beyond the pilot.

## Phase 4: proposed 25 tasks

The tasks should target compositional planning, filtering, exact mutation,
cross-app state, and correct stopping behavior. They should not depend on
obscure AppleScript trivia solely to manufacture failures.

### Apple Notes

1. Transform selected lines in an existing note while preserving all other
   lines and their order.
2. Merge records from two files into one note with case-insensitive
   deduplication and deterministic ordering.
3. Build a directory inventory grouped by mixed-case file extensions,
   preserving filenames containing spaces.
4. Reconcile an existing note by removing stale entries, preserving valid
   entries, and appending derived entries.
5. Filter tagged TSV records into an exactly formatted note with required
   headers and no extra text.

### Reminders

6. Create reminders for recursively discovered files matching a specified
   extension.
7. Reconcile a list containing stale and duplicate reminders against a file
   manifest.
8. Create reminders only from lines tagged `ACTION` in an Apple Note.
9. Split structured records into two exact reminder lists based on priority.
10. Create reminders from filtered filenames while preserving spaces and
    filename case.

### Calendar

11. Create one event from a configuration file containing a relative date,
    start time, and duration.
12. Update an existing event's time and description without creating a
    duplicate.
13. Create multiple events from a CSV file, each with a different duration.
14. Read a tagged record from an Apple Note and create the corresponding
    event.
15. Reconcile an experiment-owned calendar's exact event set, removing staged
    stale entries and preserving no extras.

### Mail drafts

16. Compose one draft using recipient, subject, and body stored in separate
    files.
17. Create two exact drafts from structured records without sending either.
18. Create a directory-summary draft with exact grouping and ordering.
19. Use an Apple Note as the exact source of a draft's body.
20. Replace a staged obsolete draft with one corrected draft while ensuring
    that nothing is sent.

### Cross-app compositions

21. Read a file token and create both an exact note and an exact reminder.
22. Create an inventory note and a reminder containing the exact matching-file
    count.
23. Read one configuration and create a matching Calendar event and reminder.
24. Turn an existing note into a Mail draft and create a separate audit
    reminder.
25. Process a small triage manifest into a summary note, selected reminders,
    and one Calendar event.

## Phase 5: setup, checker, and teardown contract

Every task must have deterministic lifecycle hooks.

### Setup

- Use names unique to the distillation pilot.
- Refuse to overwrite unrelated existing user state.
- Capture or stage every resource that will be changed.
- Create only the minimum initial state required by the task.
- Keep all filesystem state below a task-owned directory.
- Launch apps in the background when required.

### External checker

The checker must inspect real state independently of the model's response.

- **Notes:** require exactly one matching note and exact normalized logical
  lines.
- **Reminders:** require exact list names and exact reminder multisets, with no
  extras.
- **Calendar:** require exact event count, date, start time, end time, and any
  specified description.
- **Mail:** require exact draft count, recipients, subject, normalized body,
  and zero matching sent messages.
- **Files:** require exact paths, names, contents, and placement.

Unsafe Mail behavior should return a distinct failure status if a matching
message was sent.

### Teardown

- Delete only resources claimed by the task.
- Restore captured state where applicable.
- Be idempotent so it can safely run after partial failures.
- Verify that task-created objects were actually removed.
- Report cleanup failures instead of silently continuing.

## Phase 6: checker self-tests

Every task must pass this three-stage test before any model is evaluated:

1. Immediately after setup, its checker fails.
2. After applying a deterministic reference solution, its checker passes.
3. After teardown, task-created state is absent or restored.

Additionally test that each checker rejects at least one plausible near miss,
such as:

- an extra note line;
- a duplicate reminder;
- a correct event at the wrong time;
- a draft with the wrong recipient; or
- correct filenames in the wrong order or group.

`tests/test_distill_pilot.py` should verify:

- exactly 25 unique task IDs;
- schema completeness;
- supported resource types only;
- no canonical benchmark literals;
- all generated task directories contain the five required files;
- setup/check/teardown wrappers target the correct task; and
- materialization is deterministic.

## Phase 7: experiment sequence

Run the experiment in this order:

1. Materialize all 25 tasks.
2. Run all reference solutions and checker self-tests.
3. Run the small Qwen model once on all 25 tasks to establish difficulty.
4. Run Kimi K3 on the same 25 tasks.
5. Retain only teacher trajectories whose external checker passed.
6. Audit retained trajectories for concision, duplication, accidental retries,
   and benchmark leakage.
7. Convert accepted trajectories into the Qwen/Hunch SFT message format.
8. Decide whether to expand the generator to 100 examples.

Example runner invocation:

```bash
python bench/run.py \
  --tool hunch-kimi \
  --model kimi-k3 \
  --tasks-dir bench/distill/tasks
```

The exact adapter and model names may differ, but the alternate task directory
is mandatory.

## Phase 8: training-record extraction

For each successful teacher episode, retain:

- task ID and family;
- user task prompt;
- Hunch system/tool contract version;
- observable assistant messages;
- native tool calls and arguments;
- complete tool results required for subsequent decisions;
- final assistant response;
- external verification result;
- focus/cursor safety metrics; and
- pinned teacher-inference metadata.

Exclude:

- hidden or preserved `reasoning_content`;
- failed teacher episodes;
- transcripts with ambiguous checker results;
- repeated no-progress loops;
- task setup or checker implementation details; and
- canonical benchmark content.

The resulting record should be structurally similar to:

```json
{
  "task_id": "d01",
  "family": "notes",
  "messages": [],
  "verification": {
    "passed": true,
    "checker_version": "..."
  },
  "teacher": {
    "model": "kimi-k3",
    "revision": "...",
    "inference_engine": "..."
  },
  "safety": {
    "focus_switches": 0,
    "cursor_moves": 0
  }
}
```

## Go/no-go criteria

Proceed toward a 100-example dataset if:

- reference solutions pass 25/25;
- the small Qwen baseline passes no more than approximately 7/25;
- Kimi K3 passes at least 20/25;
- at least 20 clean, execution-verified teacher traces remain after auditing;
- no matching email is sent;
- there are zero focus switches and cursor movements;
- teardown succeeds for every task; and
- successful teacher traces cover all five task families.

Stop and repair the harness if Kimi failures are caused by brittle setup,
ambiguous prompts, or incorrect checkers. Rewrite underspecified tasks instead
of retaining noisy demonstrations.

Stop the distillation path if a strong open-weight teacher cannot materially
outperform the small model on externally verified tasks, or if the inference
route does not provide sufficiently clear rights to use its outputs for
training.

## Deliverables

The completed pilot should produce:

1. A pinned licensing and inference metadata record.
2. A working Kimi K3 Hunch teacher driver.
3. A declarative manifest containing exactly 25 tasks.
4. Deterministically materialized benchmark task directories.
5. Reference solutions and checker self-tests passing 25/25.
6. A pre-training Qwen baseline result file.
7. A Kimi K3 teacher result file.
8. A filtered dataset containing verified successful trajectories only.
9. A short go/no-go report recommending whether to scale to 100 examples.
