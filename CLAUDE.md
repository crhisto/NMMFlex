# NMMFlex — Claude collaboration preferences

This file captures the user's preferences for how Claude should work in
this repository. Read it at the start of every session.

## Commit and push workflow (strict)

1. **No Claude / AI attribution on commits.** Do **not** add
   `Co-Authored-By: Claude ...` trailers or any other AI attribution.
   Commits must show only the user (Crhistian Cardona) as author.
2. **Always ask before committing.** Never run `git commit` without an
   explicit "yes, commit" from the user in the current turn. A general
   earlier "you can commit when ready" does not count — re-confirm
   every time.
3. **Always ask before pushing.** Same rule for `git push`. Always ask
   first, even on feature branches.
4. **Run tests before every commit.** Before proposing a commit, run
   the project test suite and report the result. Only ask for commit
   approval if tests pass (or if the user has explicitly acknowledged
   a pre-existing failure).

   ```shell
   PYTHONPATH=NMMFlexPy/src .venv/bin/python -m pytest NMMFlexPy/tests/
   ```

5. **Every commit needs a meaningful message.** Short imperative subject
   line, then a body explaining what changed and why. No empty
   messages, no "wip", no "update".
6. **Stage explicitly.** Use named paths with `git add path/to/file`.
   Never `git add -A` or `git add .` — the gitignore may not yet cover
   everything, and accidental inclusion of `.venv/` or caches is a
   common source of noise.

## Project structure quick reference

- `NMMFlexPy/src/NMMFlex/` — Python package source.
- `NMMFlexPy/tests/` — pytest suite.
- `NMMFlexPy/benchmarks/` — performance benchmarks (not run in CI).
- `.venv/` — local virtual environment, gitignored.

## Known pre-existing issues

- `test_NMMFlex_basics::test_proportion_constraint_h_partial_fixed`
  fails on `crhisto-improved-code` against the pinned deps. Tracked
  separately — do not consider this a regression caused by new work.
