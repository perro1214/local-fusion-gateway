# Security Policy

## Supported Versions

This project is in alpha. Security fixes are handled on the `main` branch until versioned releases are established.

## Reporting a Vulnerability

Please do not open a public issue for vulnerabilities that expose secrets, private prompts, or unintended network access.

Use GitHub's private vulnerability reporting if it is enabled for the repository. If it is not enabled, contact the repository owner directly through GitHub.

## Security Expectations

- Do not commit real API keys, `.env` files, or private `config.yaml` files.
- Keep the Gateway bound to `127.0.0.1` unless you intentionally expose it.
- Treat prompts, panel outputs, and judge outputs as potentially private.
- `X-Local-Fusion-Debug: true` is intended to expose metadata only, not prompt text or model response text.
- This project does not currently provide production authentication, authorization, or tenant isolation.
