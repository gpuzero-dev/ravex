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

- **Restoring runs code from the checkpoint.** Optimizer and scheduler state
  is restored with `torch.load(..., weights_only=False)` or, under Moonclip,
  partly unpickled. Whoever can write to the checkpoint directory or bucket
  can therefore run code in the process that resumes from it. Give write
  access only to what you would let run code on your training machines.
- Ravex never reads storage credentials itself: Moonclip does, from the
  environment, a profile or the machine's role.
- Nothing is sent anywhere unless a metrics endpoint is configured
  (`RAVEX_METRICS_ENDPOINT`).
