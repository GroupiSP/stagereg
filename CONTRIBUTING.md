# Contributing to stagereg

Thanks for your interest in contributing to `stagereg`.

## Opening issues

- Open issues using one of the templates in GitHub:
  - **Bug report**
  - **Feature request**
- Choose the template that best matches your case and fill in all relevant fields.
- If you plan to implement the change, tick **"I would like to work on it"** in the issue.

## Contributing to the codebase

1. Open an issue first, and explicitly state that you would like to work on it.
2. Set up your development environment with `uv`, then install pre-commit hooks:
   - `uv sync --extra jax --group dev --group docs`
   - `uv run pre-commit install`

   The `jax` extra is needed even for core-only changes: `uv run pytest` also runs the snippets in
   `README.md` and `docs_site/`, which import JAX, Flax and Optax. The `docs` group builds the
   documentation site.
3. Open a **draft pull request** targeting the `main` branch as soon as you start the work, and
   link it to the issue you opened.
4. Implement your changes on that branch, pushing commits to the draft PR as you go.
5. Run the required checks locally:
   - `uv run pre-commit run --all-files`
   - `uv run pytest`
   - `uv run --extra jax --group docs mkdocs build --strict`
6. When the checks pass, mark the PR as ready and request review from one of the maintainers.

## Documentation

The user documentation site is built with MkDocs Material from `docs_site/`; `docs/` holds
design records and is not published. Preview it with
`uv run --extra jax --group docs mkdocs serve`.

Every fenced `python` block in `docs_site/` and `README.md` runs as a test under `uv run pytest`,
so write snippets that stand alone and finish in about a second. For a fragment that cannot run on
its own (a dataset download, a full training run), mark the fence as non-executable with the
superfences brace form, and link to the runnable example:

````markdown
```{.python notest}
result = run_tracking_example()
```
````

The form is `{.python notest}`, not `python notest`: Material's superfences does not recognise a
bare word after the language and renders the block as plain text, while pytest is configured with
`--markdown-docs-syntax=superfences` and skips brace-form fences carrying `notest`.

The API reference in `docs_site/reference/` has one page per package and one `## module`
section per source module. Exporting a symbol from `stagereg.core` or `stagereg.jax` means adding
`::: stagereg.core.Name` (its public path) under the section of the module that defines it;
`tests/test_reference_completeness.py` fails until you do, and fails on an entry that names
something not exported.

## Cutting a release (maintainers)

Releases are tag-driven by `.github/workflows/release.yml`; no local credentials are needed.

1. Bump `version` in `pyproject.toml` (and in `CITATION.cff`) in a PR, and merge it into `main`.
2. Tag the merge commit with the same version prefixed by `v`, and push the tag:
   `git tag v0.1.0 && git push origin v0.1.0`.
3. The workflow fails unless the tag matches the declared version, then runs CI (`ci.yml`) as the
   verification gate, builds the wheel and sdist, and publishes them to TestPyPI.
4. Check the TestPyPI upload, then approve the waiting **Publish to PyPI** job (the `pypi`
   environment requires a maintainer's approval). The GitHub Release, with notes generated from
   the merged PRs and the built files attached, is created after the PyPI publish succeeds.
