# Security

## Reporting a vulnerability

Please do not open a public issue. Report it privately, either through
GitHub's [private vulnerability reporting](https://github.com/gpuzero-dev/ravex/security/advisories/new)
or by email to vecchionejonathan@gmail.com.

Say what is affected, how to reproduce it, and what an attacker gains. You
will get an answer within a week, a fix or a plan for one, and credit in the
release notes if you want it.

## Supported versions

Only the latest release receives fixes. Ravex is 0.x, and a fix
is released as a new version, never backported.

## What to know before you rely on it

- **Restoring builds tensors and plain data only (GPU-538).** The
  `torch_save` backend and the checkpoints of other tools Ravex resumes from
  are read with `torch.load(..., weights_only=True)`, which also admits
  NumPy's arrays and dtypes, for NumPy's RNG state, and nothing more; under
  Moonclip, its own reader does the same (Moonclip's `SECURITY.md`).
  Anything else fails the resume with `UnsafeCheckpoint`, naming the file:
  there is no unrestricted fallback. Write access to the checkpoint
  directory or bucket still decides what a resume trains on, so give it only
  to what you trust with your runs.
- Ravex never reads storage credentials itself: Moonclip does, from the
  environment, a profile or the machine's role.
- Nothing is sent anywhere unless a metrics endpoint is configured
  (`RAVEX_METRICS_ENDPOINT`).
