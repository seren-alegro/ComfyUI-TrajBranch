# ComfyUI-TrajBranch

Branch a rectified-flow sampling run mid-flight — and resume the
second half with different settings.

Two nodes for Anima-family flow models (ComfyUI 0.44 sampler
contract, original clean-room code):

## Nodes

### Traj Branch Sampler (capture)

Runs a plain flow-Euler generation. At `capture_step` it snapshots
the latent — the branch point — and outputs all three:

- `latent` — the finished generation
- `branch_latent` — the state at the branch point
- `branch_sigma` — the noise level at the branch point

An optional `sigmas` input replaces the built-in linear-shift
grid (shift 3.0; do not stack with an AuraFlow patch).

### Traj Resume Sampler (branch)

Continues from a branch latent. Wire `branch_sigma` into
`start_sigma`, set the remaining `steps`, and the second half runs
on the tail of the original grid. The node takes its **own**
model / positive / negative / cfg, so at the branch you can change
the prompt, the CFG — even the checkpoint — and only the second
half of the trajectory feels it.

## The contract

Capture at step k, then resume with the remaining step count and
the same shift, reproduces the uninterrupted run (the resume grid
is the original grid's tail, recovered by inverting the
linear-shift parameterisation at the branch sigma; agreement is at
float32 grid precision, ~1e-7 in the sigma values — verified by a
numpy mirror test in `tests/`). Forking is therefore a controlled
act: keep the settings and nothing changes; change one and only
the tail moves.

Euler only, for now: momentum and AB2 carry velocity history,
which a branch latent alone does not hold. Resuming those exactly
would need the history exported alongside the latent.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/seren-alegro/ComfyUI-TrajBranch
# restart ComfyUI
```

Or copy this folder as `ComfyUI-TrajBranch` into
`ComfyUI/custom_nodes` manually, then restart ComfyUI.

## Example workflow

`examples/traj_branch_identity_test.json` - a minimal core-nodes-only
workflow: the Branch sampler captures at step 15 of 30 and the
Resume sampler finishes the remaining 15 steps with the same
settings, saving both results (`traj_full` / `traj_resumed`). If
capture + resume is working, the two images match. The checkpoint
name in the loader is a placeholder - point it at any Anima-family
checkpoint, and if your setup loads the text encoder / VAE from
separate files, swap that part of the chain accordingly. To fork
instead of verify, change the prompt or CFG on the Resume side only.

## Tests

`tests/test_traj.py` — numpy mirror of the grid/capture/resume
math (no torch needed): grid-tail reproduction, capture-state
identity, capture+resume vs uninterrupted run.

## License

MIT

---

## 한국어 요약

샘플링 궤적을 중간에서 분기하는 노드예요. Branch Sampler가 지정한
스텝에서 latent를 캡처하고, Resume Sampler가 그 지점부터 이어
뽑는데 — 뒤쪽 절반은 프롬프트·CFG·체크포인트까지 바꿀 수 있어요.
설정을 그대로 두면 무중단 실행과 같은 그림이 나오는 것을 수치로
검증했어요.
