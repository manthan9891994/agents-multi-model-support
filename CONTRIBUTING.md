# Contributing to dynamic-model-router

Thanks for your interest. This is a small, focused project — contributions
that align with the core goal (route each task to the cheapest model that
can handle it) are very welcome.

## Quick start

```bash
git clone https://github.com/manthan9891994/dynamic-model-router.git
cd dynamic-model-router
python -m venv .venv && source .venv/bin/activate    # or .venv\Scripts\activate on Windows
pip install -e ".[ml,dev,google,anthropic,openai]"
pytest tests/
```

## Development workflow

1. **Open an issue first** for any non-trivial change — saves both of us time if your idea collides with an in-flight effort.
2. **Branch from `master`**: `git checkout -b feat/my-change`
3. **Add tests** for any new code path. Aim for the same test density as existing modules.
4. **Run the full check**:
   ```bash
   ruff check classifier/ tests/
   ruff format --check classifier/ tests/
   mypy classifier/
   pytest tests/ --cov=classifier --cov-report=term-missing
   ```
5. **Update CHANGELOG.md** under `[Unreleased]` — one line per change.
6. **Open a PR** referencing the issue number.

## Code style

- **Be terse.** This codebase favors short, dense functions over abstractions. Three similar lines beat a premature helper.
- **No comments that restate what the code does.** Only explain *why* when it isn't obvious.
- **Prefer editing existing files** over creating new ones. Extend, don't fragment.
- **No async/await** unless wrapping sync logic — the cascade is sync at heart.
- **Keep public APIs documented** with one-paragraph docstrings + an `Example:` block.

## Adding a new layer / strategy

If you're proposing a fourth layer, please discuss in an issue first.
Layers must implement: `(task, history) -> (TaskType, TaskComplexity, ModelTier, float, str) | None`.
Return `None` for abstain — the cascade handles fallback.

## Adding a domain preset

`classifier/presets/<domain>.py` should return a dict with at minimum:
- `extra_keyword_packs`: list of `KeywordPack` instances
- `extra_pii_patterns`: list of (compiled regex, replacement-token)

See `classifier/presets/healthcare.py` for the canonical example.

## Adding an integration

Integrations live in `classifier/integrations/<framework>.py`. Convention:
- `pick_<framework>_llm(task)` — one-shot helper returning the framework's LLM type
- `Dynamic<Framework>LLM` — drop-in wrapper that classifies on each call

Mock the framework in tests so contributors don't need to install every integration.

## Reporting bugs

Use the bug report template. Always include:
- `dmr version` output
- Minimum reproducing snippet
- What you expected vs what happened

## Reporting security issues

**Do not open a public issue.** See [SECURITY.md](SECURITY.md).

## License

By contributing you agree your work will be released under the MIT license.
