# Automated GitHub Repo Factory Rules

Whenever the user asks to create, publish, or ship a new GitHub repository:

1. **Strict Pre-Commit Hygiene (Never Leak Weights or Secrets):**
   - Generate or verify `.gitignore` BEFORE `git add`. Always exclude:
     `*.pt`, `*.pth`, `*.safetensors`, `*.bin`, `*.ckpt`, `.env`, `*token*`,
     `__pycache__/`, `.pytest_cache/`, `node_modules/`, `hf_cache/`, `*.tmp`.
   - Explicitly allow tracking of core code (`smp_core.py`), tests, `starbody_studio/`,
     and cryptographic ledgers (`ledger_smp_v1.json`).
   - Run a pre-push scan (`git status --short` + check for any staged file > 25 MB
     or strings matching `ghp_`, `hf_`, `sk-`). Abort if any secret or weight file is staged.

2. **Standard Scientific / Product Scaffolding:**
   - Ensure the repo includes:
     * `README.md` (project summary, quickstart commands, hardware/precision notes,
       and current `ledger_smp_v1.json` head SHA-256 if present).
     * `pyproject.toml` (clean dependency pins and console entry points).
     * `LICENSE` (MIT for open-source metrology repos; proprietary notice for private Pro repos).
     * `.github/workflows/ci_gate.yml` running `pytest` and `python smp_core.py` on every push/PR.

3. **Automated Remote Creation & Push:**
   - Default to `--private` unless the user explicitly specifies `public`.
   - Create the GitHub repository via the connected GitHub MCP tool (or `gh repo create` fallback),
     set `main` as the default branch, push the initial commit, and return the live repo URL
     plus the commit SHA.
