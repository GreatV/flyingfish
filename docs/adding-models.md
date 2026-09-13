# Adding model adapters

The public CLI is organized by task: `text`, `video`, `music`, `3d` and
`similarity`. A checkpoint's architecture metadata selects an adapter within
that task. Checkpoint directory names do not select implementations.

Each registration in `src/cli/adapters` supplies four things:

| Field | Responsibility |
|---|---|
| `id` and `task` | Name the adapter and the task it implements |
| `recognizes` | Recognize supported architecture metadata without reading weights |
| `command` | Build the adapter's operations, typed options and defaults with Clap |
| `run` | Parse the selected operation and call its implementation |

To add another model for an existing task:

1. Implement its inference engine in a model crate, using `ff-core` for shared
   weight loading, caching and telemetry where applicable.
2. Add an adapter module with its metadata recognizer, command schema and
   execution handler. Reuse shared argument groups for common cache controls.
3. Register its descriptor in `BUILTINS` in `src/cli/adapters/mod.rs`.
4. Add checkpoint recognition and component inventory support in `src/models.rs`
   if the model should also be available to `ff models list` and `inspect`.
5. Test metadata routing, invalid options, and execution with a small fixture.

The top-level command enum and task router do not need another model branch.
The registry test `adding_a_registered_model_requires_no_task_router_changes`
demonstrates registration and dispatch of another text model.

An adapter can recognize multiple compatible architectures. The TRELLIS adapter
recognizes both TRELLIS-1 and TRELLIS.2 under `ff 3d generate`; their execution
code validates the different conditioning and resolution requirements.

General help shows options shared by models that provide the same operation.
Once a model is identified, its complete schema handles parsing, including
defaults, conflicts and required arguments. Flags belonging to another adapter
are rejected. Use `--adapter <name> --help` to inspect a particular schema before
downloading weights. An explicit adapter also resolves overlapping recognizers;
it must still support the supplied checkpoint metadata.

Parameter validation and memory admission are part of normal execution. Keep
those checks before weight materialization and output publication. They should
not become an alternate execution mode.
