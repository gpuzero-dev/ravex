# Contributing

Issues and pull requests are welcome.

- **A bug**: open an issue with the version (`ravex --version` or `pip show ravex`),
  Python and torch versions, the OS, and the smallest script that shows it.
- **A security problem**: not in an issue; see [SECURITY.md](SECURITY.md).
- **A change**: for anything beyond a small fix, open an issue first, so we
  can agree on the approach before you spend time on it.

## Making a change

```bash
pip install -e ".[dev]"
pytest tests/unit
```

`tests/unit` runs anywhere; `tests/distributed` needs Linux (see the README's
Development section).

- Keep a pull request to one change, with a test that fails without it.
- Code, docstrings and comments say *why*, in full sentences. Match the
  style of the file you are in.
- Add a line to the `Unreleased` section of `CHANGELOG.md` saying what
  changed for a user.

By contributing you agree that your contribution is licensed under the
Apache License 2.0, as the rest of the project.
