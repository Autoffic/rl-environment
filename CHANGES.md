# RL Traffic-Light Environment — 2026 Change Log & Design Notes

This document records *every* change made in 2026 (the last commits before it
are from 2024-08-05). It covers both the one committed 2026 change and the
entire uncommitted working-tree delta, plus the experiments run, why each
change was made, and how it was verified. Read it top-to-bottom to fully
understand the current state before training again.

Extra context that applies everywhere:

- The model runs are the A/B experiments themselves; most of 2026 was spent
  chasing a training collapse (see §5).
- Nothing except the `2026-08-14` commit is committed. `git status` shows a
  large diff across the env, route generator, train/traffic scripts, plus
  several new untracked files (`CHANGES.md`, `scripts/sweep.py`,
  `scripts/generate_test_routes.py`, `sumo-files/test-routes/`,
  `sweep_results/`, and the many `models/.../` artefacts).

---

## 1. Timeline of 2026 work

| when | what |
|---|---|
| **2026-08-14** | Commit `b671edd` "fix: try to fix logical errors (WIP)" — Gymnasium API + logic fixes (see §2). |
| 2026-08-14/15 | Explored tooling: many short `models/` runs; obs redesign experiments. |
| 2026-08-16 18:16 | **AUG18** — obs-14 (no elapsed green), no curriculum. The last *good* run: eval reward **−593**, entropy **0.788**, clip_fraction **0.03**, mean waiting 52→19.5 s. Saved as best_model; now the `traffic.py` default. |
| 2026-08-16 22:50 | **AUG22** — obs-15 + curriculum, 1.78M steps: **collapse** (uniform policy, EV~0). |
| 2026-09-19 | **SEP19** series — obs-15 + curriculum again: **collapse** in every run (uniform policy). Root-cause analysis happened here (see §5). |
| 2026-09-19 20:43 | Full 2M-step run launched with the fixes, then **interrupted on request** → `exit_before_finished.zip` saved; `sweep.py NEW_MODEL` points here. |
| (recent) | Smoke training verified the fixes (§6); two secondary bugs fixed (§4.6, §2.2). |

The 2026 model runs in `models/` (all before the fixes):
`2026-08-14`, `2026-08-15` (many), `2026-08-16 15:12`, `18:16` (AUG18),
`21:42`, `22:45`, `22:50` (AUG22), `2026-09-19 19:04` + `20:43/47/53/58`
(SEP19 + early retries).

---

## 2. COMMITTED — `b671edd` (2026-08-14)

`git show b671edd` — "fix: try to fix logical errors (WIP)". Files touched:
`README.md`, env, `scripts/traffic.py`, `scripts/train.py`, `setup.sh`.

### 2.1 Env — Gymnasium API / logical fixes
- `step()` now returns proper `terminated`, `truncated` instead of a single
  `done`; no longer calls `self.reset()` inside `step()` (episodes end, SB3
  resets).
- `reset()` no longer conditionally adds the step listener — `addStepListener`
  is always called after `traci.start`.
- Removed the "clear vehicle ids" call from `step` before reward computation
  (kept only after reward calc).
- `calculate_stopped_percentage()` now counts vehicles with `speed < 0.1`
  (`< 0.36 km/h`) over `vehicle_ids_between_steps`, instead of the broken
  `getStopState(veh) == 1` count over the total (buggy denominator).
- `generate_routefile(...)` call made explicit with keyword args.
- Cosmetic: quote style, formatting.

### 2.2 train.py
- New constants `N_TRAIN_ENVS = 8`, `N_EVAL_ENVS = 2`, and named hyperparams
  block: `VERBOSE=1, BATCH_SIZE=64, N_STEPS=256, N_EPOCHS=10, ENT_COEFF=0.1,
  GAMMA=0.8138`.
- `n_envs = min(n_envs, cpu_count() - 1)` (leave a core free).
- Training/eval envs now use `N_TRAIN_ENVS`/`N_EVAL_ENVS`.

### 2.3 traffic.py
- Default `model_path` bumped from the 2024-07-31 model to **2024-08-05
  21:20** best_model.zip.

### 2.4 setup.sh
- `set -x`/`set +x` trace around the export, and `SUM0_HOME` value quoted.

### 2.5 README
- New "Setup Environment" (`source ./setup.sh`) section; updated run commands
  to `./scripts/...`.

---

## 3. UNCOMMITTED — route generator (`generateRouteFile.py`)

All this is uncommitted 2026 work (nothing here existed on 2024-08-05).

### 3.1 Regime / density / curriculum profile system
- `REGIME_PROFILES` — per-approach demand weights (E0, -E1, -E2, -E3):
  - `uniform` → `[0.25, 0.25, 0.25, 0.25]`
  - `optimal` → `[0.40, 0.20, 0.25, 0.15]`
  - `very_optimal` → `[0.70, 0.10, 0.12, 0.08]`
- `DENSITIES` → `[0.1, 0.2, 0.4]` vehicles/sec total.
- `CURRICULUM_START_WEIGHTS` = very_optimal, `CURRICULUM_END_WEIGHTS` =
  uniform, `CURRICULUM_START_DENSITY = 0.05`, `CURRICULUM_END_DENSITY = 0.4`,
  `CURRICULUM_RAMP = 0.1` (within-episode ±10%), `CURRICULUM_SEGMENTS = 40`.
- `pick_profile(seed)` — deterministic (seed % 9) selection over the
  regime-major 3×3 grid.
- `curriculum_profile(progress)` — linear interpolation (Lunar-Lander style)
  of weights and density from the easy anchor (low + imbalanced) to the hard
  anchor (high + uniform).

### 3.2 Deterministic, sized generation
- `generate_routefile(...)` gains `seed, regime, density, weights, curriculum,
  base_weights, base_density, num_segments` kwargs.
- `random.seed(seed)` applied **twice**: once before the filename pick (so the
  filename-selection loop can't shift the RNG used for content), once right
  before content is emitted.
- `vehicle_count(weight, seg_begin, seg_end, vps)` — count scales linearly
  with horizon (so per-second density is identical for 79k training vs 1k eval
  routes), multiplies by `U(0.5, 1.5)`, floor at 1 vehicle.
- Non-curriculum path uses `horizon * density * weight[approach]` per approach.

### 3.3 Exact split semantics + no zero-count flows
- `split_approach(total, shares)` — **largest-remainder** allocation for the
  `[0.33, 0.33, 0.34]` movement splits, so counts sum *exactly* to the
  approach total (old code used three independent `int()` calls that silently
  lost vehicles).
- `interval_xml(...)` + `flow(...)` helper — each of the 12 movement flows is
  emitted **only when its count ≥ 1**. A `number="0"` flow made SUMO warn
  "Flow 'x' has no instances". Retrieved/regex consumers never parse flow-ID
  prefixes, so dropping zero-count flows is safe.
- Curriculum mode splits the horizon into `num_segments` intervals whose
  density swings `±10%` around the episode base (a light rush-hour ramp).

---

## 4. UNCOMMITTED — env, train, traffic, and the new tools

### 4.1 Env observation → normalized, obs-14 / obs-15

`_build_observation()` replaces the raw 12 lane counts:

| index | meaning |
|---|---|
| 0–11 | per-lane **fraction** of the observed total (scale-invariant shape — decouples the distribution from absolute load) |
| 12 | raw total vehicle count on observed lanes |
| 13 | green phase index `getPhase() // 2 ∈ {0..3}` |
| 14 | *(optional)* elapsed green time, seconds |

- `include_elapsed_green=True` → obs size 15; `False` → 14. This is the
  obs-15 vs obs-14 A/B.

### 4.2 Env seeding
- Env takes `seed: int | None`. Each `reset()` uses `route_seed = seed +
  reset_count` (varies across episodes, fully reproducible sequence).
- `route_seed` drives **both** `generate_routefile(seed=...)` and SUMO's
  `--seed`, so route + simulator RNG are locked together.

### 4.3 Env curriculum / mixed-regime wiring
- `mixed_regimes=True`: each reset picks `(regime, density) = pick_profile(seed)`
  → random exploration across the 9-grid difficulty set.
- `curriculum=True`: reads shared progress via `curriculum_progress` (a
  `multiprocessing.Value` in training, plain float elsewhere) and maps it to
  `(base_weights, base_density) = curriculum_profile(progress)`.
- **Deliberate constraint:** curriculum progress is NOT in the observation —
  it's a real-world-unavailable quantity. The policy must generalize from
  observable state alone.

### 4.4 Env reward pipeline (the root-cause fix) — see §5 for the why

`calculate_reward()` / `composite_reward()` now uses:

| term | default | effect |
|---|---|---|
| `+reward_on_throughput_and_waiting` | scale 100 · (passing×1.0 − waiting×0.8) | dense ±~100 throughput signal |
| `−switch_penalty` | 15.0 | discourage idle switching |
| `−forced_switch_penalty` | 5.0 | the safety override, mild |
| `−anti_starvation_penalty × (max_wait − 60)` | 0.25 | only for waits > 60 s |
| `−queue_pressure_penalty × stopped_fraction` | 0.2 | bounded in [0,1], density-invariant (fraction of stopped vehicles, not a raw count) |
| `−early_truncation_penalty` | 50.0 | on >80% stopped reset |
| hard clip `reward_clip` | 200.0 | ±200 bound; `None`/0 disables |

Removed / fixed:
- The **quadratic excess-wait** term `0.01 × mean((w − 60)²)` and the absolute
  `0.2 × stopped_count` (both blew reward to −110…−550 and value loss to
  ~2e7–3e7).
- Old `reward_on_decrease_in_waiting_time` (a spiky *difference* of cumulative
  waiting) is no longer the main reward.

New metrics (`info`): `forced_switch`, `max_grn_duration`,
`max_grn_phase_durn` (per phase), `early_truncation`.

### 4.5 Env phase logic
- Holding a phase past `max_green` (120 s default) is forbidden — `step()`
  forces a switch to the next green phase (`forced_switch=True`, penalty 5.0).
- `elapsed_green` tracked: `min_green` after a switch, `+= min_green` when held.
- Per-episode max green duration tracked per phase for diagnostics/green-duration eval.

### 4.6 train.py — training redesign

- Constants: `TOTAL_TIMESTEPS_FOR_SUMO = 20_000` (was 79_000; short episodes →
  less difficulty mixing per rollout, critic tracks curriculum), `MODEL =
  2_000_000`, `CURRICULUM=True`, `UPDATES_PER_STAGE = 1` (ramp difficulty every
  PPO update — the smoothest possible ramp), `REWARD_CLIP = 200.0`,
  `INCLUDE_ELAPSED_GREEN = True`.
- Hyperparams: `BATCH_SIZE=256`, `N_STEPS=2048`, `GAMMA=0.99` (was 64/256/
  0.8138). Eval: `EVAL_FREQ=N_STEPS`, `EVAL_EPISODES=20`,
  `deterministic=True`.
- **VecNormalize on the training env**: `norm_obs=False`,
  `norm_reward=True`, `clip_reward=10.0`, `gamma=GAMMA`. Reward statistics are
  normalized per-env so the shared critic isn't dominated by one env's
  difficulty; observations stay raw (directly deployable).
- **CurriculumCallback**: a shared `multiprocessing.Value` that starts at `0.0`
  and advances `progress = updates / total_updates` every `UPDATES_PER_STAGE`
  updates. Env resets (fork subprocesses) read it → between-episode difficulty
  ramps with training time, not with performance.
- `get_env(...)` now forwards the whole reward-knob set and `base_seed`
  (each training env gets `seed = base_seed + index`).
- **Eval env is VecNormalize-wrapped too** — *the crash fix*. SB3's
  `EvalCallback.sync_envs_normalization` asserts the eval env is also a
  `VecNormalize`; a plain `SubprocVecEnv` eval raises `AssertionError` at the
  first evaluation. Fix: `VecNormalize(eval_env, training=False, norm_obs=False,
  norm_reward=False, gamma=GAMMA)` with `base_seed=42`. `norm_reward=False`
  keeps eval rewards raw → `eval/mean_reward` stays comparable across runs.
- `save_on_exit` SIGINT handler saves the model before exiting (graceful stop).

### 4.7 traffic.py — inference harness

- **Lazy model load**: `load_model(path)` sets a module-level `model`. Called
  *inside* the subprocess so `--model-path` (set in the parent) is honored
  under forkserver/spawn where the module is re-imported (the old code loaded
  the module-default path unconditionally).
- **Observation-dim aware**: reads `model.observation_space.shape[0]`; for
  obs ≥ 14 it appends phase index + elapsed green to the lane counts, tracking
  `elapsed_green` across `held_phase` in the inference loop. Old 12-dim models
  keep working.
- **New output**: `*.green-duration.csv` — per-phase `max_green_duration` and
  `switch_count`, via a `StepListener` tuned with `junction=` that chases
  phase transitions.
- `run(...)` gains `total_timesteps` (horizon override); `start_logging(...)`
  gains `model_path` and `route_file_path`.

### 4.8 New files

- **`scripts/generate_test_routes.py`** — builds the **9 static benchmark
  routes** at `sumo-files/test-routes/<regime>-<density>.rou.xml`
  (uniform/optimal/very_optimal × 0.1/0.2/0.4), fixed seed 42, horizon 2000 s,
  same MOVEMENTS/SPLITS machinery as the training generator.
- **`scripts/sweep.py`** — head-to-head benchmark: `no_rl` (fixed cycle) vs
  `OLD_MODEL` vs `NEW_MODEL` across the 9 route files, simulator horizon 2000 s,
  collecting mean waiting time + green-duration + switch counts → writes
  `sweep_results/{results,green_duration}.csv` + `sweep_plot.png`.
  `NEW_MODEL` currently → `models/2026-09-19 20_43_50…/best_model.zip`
  (the interrupted run).
- **`sumo-files/test-routes/`** (the 9 generated `.rou.xml`),
  **`sweep_results/`** (with an `archive/`).

### 4.9 The `random-route-triple.rou.xml` diff
Touch-file regenerated by the env; content changes are an artifact of the new
generator (sized counts, exact splits, no zero flows). Not a functional change
beyond what §3 describes.

---

## 5. The collapse — analysis that drove the fixes

**Symptom** (AUG22 + SEP19): stable uniform policy — entropy pinned at
`−ln(4) = −1.386`, `clip_fraction = 0` (no update ever trusted an advantage),
`explained_variance ≈ 0`, `value_loss ~2e7–3e7`, per-action reward −110…−550
(vs −32…−70 for learned runs), mean waiting flat ~45 s.

**Root causes:**

1. **Reward-scale explosion.** The quadratic excess-wait term
   (`0.01 × mean((w−60)²)`) plus `0.2 × stopped_count` produced huge,
   occasionally extreme, per-action rewards. Advantages were dominated by
   noise; every PPO update rejected its own policy change
   (`clip_fraction = 0`). The critic couldn't learn a stable target → policy
   neurons dead on arrival. → Fixed with the reward pipeline in §4.4
   (bounded, smooth, density-invariant) **and** VecNormalize reward
   normalization (§4.6).
2. **Curriculum non-stationarity across async envs.** 8 envs each drew an
   independent difficulty at reset; the shared value function had to predict
   wildly different difficulty levels in the same rollout → critic can never
   converge. Combined with short-horizon + `UPDATES_PER_STAGE=1` (§4.6) the
   difficulty now rises slowly and *synchronously* with training time.
3. **obs-15 (elapsed green) is correlated but NOT proven causal** — the good
   run was obs-14 + no curriculum; both failures were obs-15 + curriculum. It
   stays as an A/B knob.

Observations:
- Guideline used: reward terms must be scale-stable across density and bounded
  (hard clip ±200) so advantage estimation stays healthy.
- Contrast pair used throughout: the learned run (AUG18: −593 eval, entropy
  0.788, clip 0.03, wait 52→19.5 s) vs the collapsed runs.

---

## 6. Verification performed

- **Smoke training** (obs-15, curriculum, 12k steps, 2 envs): `value_loss`
  0.972 → 0.056 over ~5 updates (failed runs ~2e7); `clip_fraction` 0.012–0.034
  (failed runs 0 exactly); healthy `approx_kl`; `entropy_loss` still −1.386 and
  EV slightly negative — **expected** at 5 updates (the uniform baseline still
  pays switch/penalty costs; the critic must become predictive first). The
  decisive evidence is that value loss shrank to a learnable scale and updates
  are accepted.
- **Eval-sync fix**: no "not of the same type VecNormalize" warning;
  `sync_envs_normalization` succeeds; train rewards verified normalized within
  ±10.
- **Flow fix**: worst-case reproduction (curriculum at progress 0, density
  0.05, 12 segments) → 72 flows, **0** with `number="0"`, 12 flow families
  preserved, 95 vehicles total conserved; SUMO loads the route with **0** "no
  instances" warnings.
- **Special-process constraint**: two env instances in one process collide on
  the traci label `conn_0`; obs-14 vs obs-15 (or any two-env comparison) must
  run in separate processes.

---

## 7. Current status

- Full 2M training launched 2026-09-19 20:43:50 (pid 3518), obs-15 +
  curriculum + VecNormalize, then **interrupted on request** → graceful save
  `exit_before_finished.zip` (67 KB) in that run dir; all processes cleaned.
- That run predates the two §4.6/§3.3 fixes (eval-sync crash and no-zero
  flows), so its log is not a valid verdict — **restart from `scripts/train.py`
  to pick those up**.
- `scripts/sweep.py NEW_MODEL` is wired to the interrupted run's
  `best_model.zip` path; update after the next real training run.

---

## 8. A/B knobs (quick reference)

| knob | file | meaning |
|---|---|---|
| `include_elapsed_green` | env `__init__` | obs-15 vs obs-14 regression |
| `CURRICULUM` | train.py | on/off difficulty curriculum |
| `UPDATES_PER_STAGE` | train.py | curriculum ramp granularity |
| `MIXED_REGIMES` | env `__init__` | random regime/density per reset |
| `REWARD_CLIP` | env `__init__` | ±200 hard bound; `None`/0 disables |
| `reward_scale`, passing/waiting weights | env `__init__` | balance throughput vs waiting |
| `--model-path` | train.py / traffic.py | which checkpoint to load |

---

## 9. Next steps

1. Restart full training with the current code (picks up the eval-sync and
   flow fixes).
2. Monitor the diagnostic trio every ~50k steps: `entropy_loss` falling off
   uniform, `clip_fraction > 0`, `value_loss` at ~1e-1..1e0 scale, EV moving
   positive, waiting time dropping.
3. On a good model: repoint `scripts/sweep.py NEW_MODEL`, run the 9-cell
   head-to-head, compare mean waiting / green durations against the fixed
   cycle and the 2026-08-16 baseline.