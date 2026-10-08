# Build and publish documentation

The handbook uses [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) and [static-i18n](https://ultrabug.github.io/mkdocs-static-i18n/setup/choosing-the-structure/). Chinese `.md` pages pair with `.en.md` translations in the same directory. Chinese is served at the site root and English at `/en/`. The language switcher opens the corresponding page; navigation and search support both languages.

## Preview locally

Documentation does not require LLVM, the Intent compiler or backend SDKs:

```bash
python3.11 -m venv /path/outside-checkout/docs-venv
source /path/outside-checkout/docs-venv/bin/activate
python environment/docs.py install
python environment/docs.py serve
```

Open the printed `http://127.0.0.1:8000/Intent/` address; the root redirects to this project path. Dependencies come only from the `docs` extra in `pyproject.toml`, with no second requirements list. Python 3.10 can also run the helper after installing `tomli`; Python 3.11+ uses the standard library.

Build static files:

```bash
python environment/docs.py build
python environment/docs.py build --site-dir /path/outside-checkout/site
```

The default output is `$XDG_CACHE_HOME/intentdsl/docs/site` or `~/.cache/intentdsl/docs/site`. The helper uses strict builds and rejects output inside the checkout. Do not commit generated files.

## GitHub Pages

`.github/workflows/documentation.yml` builds both languages on relevant pull requests. Main pushes and manual runs upload the static artifact and deploy it through GitHub's official Pages actions. A repository administrator must select **Settings → Pages → Build and deployment → Source → GitHub Actions**. The deployment output gives the actual URL. The current configuration targets `https://tianchen-ai.github.io/Intent/`, with English under `/en/`.

A local build does not mean the site is deployed. When moving the repository or using a custom domain, update `site_url`, `repo_url` in `mkdocs.yml` and README links. See [GitHub custom Pages workflows](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages) for deployment details.

## Edit pages

Usage instructions belong in `getting-started/`; stable programming-model, language and compiler specifications remain in their directories; contribution and extension instructions belong in `development/`. Maintain Chinese and English together, keeping identifiers, parameter names, complete configurations and numerical bounds aligned. Presentation and translation do not change author semantics.

Source links point to GitHub; handbook pages use relative `.md` links. Every handbook page has a matching English translation, including compiler development entry points. Missing translations do not fall back to Chinese. After editing, run the strict build and check navigation, links and code snippets in both languages.

## Publish a package preview

`.github/workflows/distribution.yml` builds installation artifacts. Ordinary pushes and pull requests only build and check them. A maintainer can manually run Distribution with `publish-preview` selected to update the public `preview` assets and tag from artifacts that passed the same build and isolated installation checks. README and installation instructions link to that public download page.

GitHub preview and PyPI are separate publication channels; uploading GitHub assets does not make `pip install intentdsl` available.

## Configure PyPI publication

Distribution's separate manual `publish-pypi` option authenticates with a PyPI API token. A repository administrator configures repository secret `PYPI_API_TOKEN` under GitHub **Settings → Secrets and variables → Actions**. The workflow reads that secret; tokens do not belong in source, packages or logs.

Manually run Distribution on GitHub `main` with `publish-pypi` selected. Its `pypi` job uploads the sdist and wheel that passed that run's build and isolated installation checks to PyPI project `intentdsl`. This option is independent of `publish-preview`: update either channel or both.

Ordinary pushes and pull requests do not upload packages. Check that run's publication job and the PyPI project page to confirm an actual successful upload. A GitHub token cannot replace PyPI credentials, and a successful build does not establish availability from the package index.
