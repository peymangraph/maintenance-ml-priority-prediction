# Security and Credential Policy

This repository is intended to be public. **Never commit credentials, secrets, tokens, passwords, private keys, connection strings, or customer-sensitive data.**

## Never commit

- API keys or access tokens
- `.env` files containing secrets
- Database usernames/passwords or production connection strings
- SSH/private keys, certificates, or signing keys
- Cloud service credentials or service-account files
- GitHub personal access tokens
- Real customer maintenance records or personally identifiable information

## Safe pattern

Use environment variables for secrets and keep only a sanitized `.env.example` in the repository.

Before every push, review staged changes with:

```bash
git diff --cached
```

Also check repository status with:

```bash
git status
```

If a secret is ever committed, do not merely delete the file in a later commit. Treat the secret as exposed, revoke/rotate it immediately, and then remove it from Git history if necessary.

## Synthetic data

The project dataset should remain synthetic unless a future data-governance review explicitly approves another source. Real customer or production maintenance data must not be added to this public repository.
