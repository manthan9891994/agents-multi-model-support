# Security Policy

## Supported Versions

Only the latest minor version receives security patches.

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

**Please do not file a public GitHub issue for security vulnerabilities.**

Instead, email **manthansinhvaghela@gmail.com** with:
- A description of the issue
- Steps to reproduce
- The affected version(s)
- Your assessment of impact

You should expect a response within **5 business days** acknowledging receipt.
A fix or mitigation timeline will follow within **30 days** for confirmed issues.

If you do not receive a response within 5 business days, please follow up — it
may have been caught in spam.

## Disclosure Policy

We follow coordinated disclosure:
1. You report the issue privately.
2. We confirm and develop a fix.
3. We release the fix and credit you (with your permission) in the release notes.
4. Public disclosure happens only after the fix is published, typically 30–90 days after the initial report.

## Scope

In scope:
- The `dynamic-model-router` package code
- The `dmr` CLI
- GitHub Actions workflows in this repo

Out of scope:
- Vulnerabilities in upstream dependencies (report to the dependency directly)
- Issues requiring local code execution to exploit (e.g., loading attacker-controlled YAML configs)
- API quota / rate-limit abuse in the underlying model providers

## Known Risks

- **User-supplied regex patterns**: Custom PII patterns passed to `Router(extra_pii_patterns=...)` are validated for catastrophic backtracking, but the validation is best-effort. Treat untrusted patterns the same way you'd treat untrusted code.
- **Layer 2 PII scrubbing is not a HIPAA certification**: The scrubber removes common PHI patterns but is not a substitute for a Business Associate Agreement with your model provider.
- **Decision logs may contain PHI**: If you enable `log_decisions=True` in a healthcare deployment, ensure your logging pipeline is HIPAA-compliant.
