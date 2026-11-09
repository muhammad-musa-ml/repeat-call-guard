# repeat-call-guard

repeat-call-guard finds tool calls that gave an agent run nothing new, labels how a failed run ended, and holds back a call that repeats one whose result the run already has. It reads recorded rollouts (JSON Lines) and trace spans shaped like Sim's `TraceSpan` type, and it can run a small retrieval task against a local Ollama model with and without the guard.

## Why

For [agent-distillation-lab](https://github.com/muhammad-musa-ml/agent-distillation-lab) I recorded 80 rollouts of a local Qwen3 4B agent on a search task. 23 of them failed. I wanted to know where each failure came from: the retrieval, the tools, or the model. Counting repeated calls gave a first answer for this data, set out below.

## Run it

Python 3.11 or newer, standard library only; `pytest` for the tests. From the repository root:

```
python -m pytest -q
PYTHONPATH=src python -m repeatguard.cli analyze data/llm_teacher_rollouts.jsonl --runs
```

`live` needs [Ollama](https://ollama.com/) serving `qwen3:4b-instruct-2507-q4_K_M`.

## What a repeat is

A step repeats an earlier one when it is the same call with the same arguments and returns the same result as that call's last usable result. A retry after an error or an empty result is not a repeat, because that try gave the run nothing to build on; an error or an empty result in between does not erase the earlier usable result. A step whose content is unknown is never judged.

A failed run is labelled by its last step: `execution` (it ended on a tool error), `unknown_end` (its content is missing), `retrieval` (it ended on an empty result), `model_repeat` (none of those, and it had repeated steps) or `budget` (none of those, and no repeats). A failed run with no calls is `no_calls`, and a run with no recorded outcome is `unlabelled`. These are rules over the trace. They say where a run stopped, not why.

## The 80 recorded rollouts

`data/llm_teacher_rollouts.jsonl` is copied unchanged from agent-distillation-lab (the git blob id matches). `results/rollouts-analysis.txt` lists every run.

```
runs: 80 (succeeded 57, failed 23, unlabelled 0)
steps: 565; repeated steps: 175 (31.0%)
succeeded runs with a repeated step: 43 of 57
failed runs with a repeated step: 23 of 23
failed runs: 184 steps, 121 of them repeated (65.8%)
how the failed runs ended: model_repeat 23
```

Every repeated step in the file is a search: 87 of `search_org` and 88 of `search_project_alias`. All 23 failed runs stopped at the 8-step cap. In each one the prompt for the last step showed a search hit that had not been opened yet; in two of them (P-101-31 and P-101-47) the last step then opened it, one step short of `answer`. The system prompt for these runs already says "Do not repeat a search that has already returned a hit". No failed run ended on a tool error or an empty result.

Of the 80 tasks, 29 had a primary search that comes back empty (18 for the project, 18 for the organization), and 14 of the 23 failures were among them.

## Sim trace spans

`repeatguard analyze FILE --spans` reads a JSON file holding a span list, `{"traceSpans": [...]}`, or a log record with `executionData.traceSpans`. The shape comes from the `TraceSpan` type in `apps/sim/lib/logs/types.ts` of simstudioai/sim at commit 525cd52 (v0.9.50), and the span builder next to it in `lib/logs/execution/trace-spans/factory.ts`: a tool call is a child span with `type: "tool"`, arguments in `input`, the result in `output`, and `status: "error"` when it failed. Sim stores a result that is not an object as `{"value": result}`, which is read as empty when the value is empty. A span with no `output`, or with `contentWithheld` set, is never judged a repeat. Old spans that keep tool calls in `toolCalls` are not read. `examples/spans.json` is a file I wrote to that shape.

## The guard and the live comparison

`RepeatGuard` remembers what each call last returned and refuses to run the same call again while that result was usable. `repeatguard live` runs the retrieval task from agent-distillation-lab against the local model in three arms with the same 8-step cap:

- `baseline`: every call the model picks is run.
- `withhold`: a repeat is not run, the history tells the model "not run: ... already returned ...", and the step still counts against the cap.
- `mask`: as `withhold`, and the withheld call is removed from the model's action list for the rest of the run.

The model decides at temperature 0 with its output limited to the allowed action names.

I used seed 211, a task set I had not looked at when I wrote the guard, with a 25% miss rate. I planned 20 tasks and stopped after 10, because each task took about 4 to 5 minutes on a machine shared with other work. `results/live-seed211.jsonl` has one row per task and arm; `repeatguard compare results/live-seed211.jsonl` produced `results/live-comparison.txt`:

```
arm       tasks  succeeded  rate    tool calls  decisions  withheld
baseline     10          5   50.0%        7.20       7.70      0.00
withhold     10          7   70.0%        4.10       7.40      2.60
mask         10          8   80.0%        5.50       7.30      1.00
withhold against baseline, same 10 tasks: both 5, only withhold 2, only baseline 0, neither 3; exact sign test p = 0.5
mask against baseline, same 10 tasks: both 5, only mask 3, only baseline 0, neither 2; exact sign test p = 0.25
```

Tool calls and decisions are means per task. A withheld call is a decision that runs no tool, so the cap is spent either way.

What the rows show:

- With 10 tasks, neither difference in success is large enough to rule out chance.
- In the 7 tasks where the first project search found the project, baseline succeeded in 5, withhold in 6 and mask in 7. The two baseline failures (P-211-1 and P-211-4) kept choosing `search_org` after a hit. With `mask`, both went on to `search_org_alias`, `open_org` and `answer`; with `withhold`, one did.
- In the 3 tasks where the first project search came back empty (P-211-0, P-211-3, P-211-7), baseline succeeded in none and withhold and mask in one (P-211-7). In P-211-0 and P-211-3 the model got a result from `search_project_alias` and then, with `withhold`, chose it six more times (all withheld) or, with `mask`, went back to the empty `search_project` five times. It never opened the project. The guard stops a repeat; it does not tell the model what to do instead.
- Withhold failed one task that mask passed (P-211-4): the model chose `search_org` five more times and each was withheld.

## What this does not show

- One model, one synthetic task and one tool set. The tasks come from the generator in agent-distillation-lab with a 25% miss rate.
- The guard enforces a rule the prompt already states. It shows that a harness can hold a model to that rule, not that the model would follow a rule it was never given.
- An exact-repeat rule is wrong for a tool that is meant to be polled. The guard is for calls whose result cannot change.
- Nothing here was run on traces exported from a running Sim; the span tests use spans written to the type.
- The live comparison is 10 tasks, run once per arm. I did not run the set twice, so I do not know the run-to-run spread of a model at temperature 0, and I did not run the live arms on the 80 recorded tasks.
- The recorded rollouts and the live runs use different task sets, so the live baseline is not a replay of the recorded runs.
