# 文档构建与发布

工具书使用 [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) 和 [static-i18n](https://ultrabug.github.io/mkdocs-static-i18n/setup/choosing-the-structure/) 构建。中文 `.md` 与同目录英文 `.en.md` 一一对应，中文在站点根路径，英文在 `/en/`。顶部语言开关切换当前对应页；导航和搜索支持两种语言。

## 本地预览

只构建文档不需要 LLVM、Intent 编译器或后端 SDK：

```bash
python3.11 -m venv /path/outside-checkout/docs-venv
source /path/outside-checkout/docs-venv/bin/activate
python environment/docs.py install
python environment/docs.py serve
```

打开终端输出的 `http://127.0.0.1:8000/Intent/`；根地址会重定向到项目路径。依赖只从 `pyproject.toml` 的 `docs` extra 读取，不建立第二份 requirements。Python 3.10 也可使用脚本，读取 TOML 需要先安装 `tomli`；Python 3.11+ 使用标准库。

构建静态文件：

```bash
python environment/docs.py build
python environment/docs.py build --site-dir /path/outside-checkout/site
```

默认输出为 `$XDG_CACHE_HOME/intentdsl/docs/site` 或 `~/.cache/intentdsl/docs/site`。脚本使用 strict build，拒绝仓库内输出；构建结果不提交 Git。

## GitHub Pages

`.github/workflows/documentation.yml` 在相关 pull request 中构建双语站点；main 推送和手动运行则上传静态 artifact，并通过 GitHub 官方 Pages actions 部署。仓库管理员需要在 **Settings → Pages → Build and deployment → Source** 选择 **GitHub Actions**。实际发布地址是 workflow 的 deployment output；当前配置对应 `https://tianchen-ai.github.io/Intent/`，英文路径为 `/en/`。

本地构建不表示网站已部署。移动仓库或使用自定义域名时，修改 `mkdocs.yml` 的 `site_url`、`repo_url` 与 README 链接。部署机制见 [GitHub 自定义 Pages workflow](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)。

## 修改页面

使用说明放在 `getting-started/`，稳定编程模型/语言/编译器规格保留各自目录，贡献与扩展说明放在 `development/`。同一改动同时维护中文和英文；代码标识符、参数名、完整配置和数值边界保持一致。规格展示与翻译不改变作者语义。

源码链接指向 GitHub，站内文档使用相对 `.md` 链接。每个工具书页面都维护同名英文页，包括编译器开发入口；缺少翻译不回退为中文。修改后运行 strict build，确认两种语言的导航、链接与代码片段。

## 发布安装包预览

安装包由 `.github/workflows/distribution.yml` 构建。普通 push 和 pull request 只构建与检查；维护者手动运行 Distribution，并勾选 `publish-preview`，才会用通过同一套构建和隔离安装检查的产物更新公开 `preview` 附件与标签。README 和安装指南链接到这个公开下载页。

GitHub preview 与 PyPI 是不同发布渠道；上传 GitHub 附件不会让 `pip install intentdsl` 自动可用。

## 配置 PyPI 发布

Distribution 的另一个手动选项 `publish-pypi` 使用 PyPI Trusted Publishing，不需要在仓库保存长期上传 token。首次发布前，PyPI 项目维护者需在 [pending publisher 页面](https://pypi.org/manage/account/publishing/) 配置：

| 字段 | 值 |
|---|---|
| PyPI 项目 | `intentdsl` |
| GitHub owner | `Tianchen-ai` |
| Repository | `Intent` |
| Workflow | `distribution.yml` |
| Environment | `pypi` |

配置完成后，在 GitHub 的 `main` 手动运行 Distribution 并勾选 `publish-pypi`；发布 job 使用该轮已通过检查的 sdist 和 wheel。项目首次创建与 OIDC 配置见 [PyPI 官方指南](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)。只有 PyPI 实际上传成功后，才能把安装入口改为 `pip install intentdsl`。
