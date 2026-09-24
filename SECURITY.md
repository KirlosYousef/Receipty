# Security

## Provider API keys

Receipty reads `OPENROUTER_API_KEY` at runtime. Local Python reads it from the
gitignored `.env` file or the process environment. Docker Compose mounts the
ignored `.secrets/openrouter_api_key` file at
`/run/secrets/openrouter_api_key`; Pydantic reads that file. Pydantic holds the
value in `SecretStr` and the application unwraps it when constructing an
OpenRouter client. Do not print, serialize, trace, or return the unwrapped
value.

`SecretStr` masks routine display and JSON serialization. It does not encrypt
the value or remove it from process memory. The OpenRouter client needs the
real key to authenticate requests. Someone who controls the host or application
process can still read it. Compose secrets reduce exposure through container
environment inspection and grant the secret only to the API service. The local
source file still needs appropriate filesystem protection.

Treat configuration and secrets differently. A model name or timeout can be
committed because it describes behavior. An API key grants access to an
external account and must stay outside the repository.

### Storage by environment

- **Local development:** for direct Python runs, put the key in the gitignored
  `.env` file. For Compose, put the key alone in the ignored
  `.secrets/openrouter_api_key` file. Compose mounts it only into the API
  container. `.dockerignore` excludes both files from the image build context.
- **CI:** the standard quality workflow does not need a live provider key. If a
  separate live evaluation workflow is added, store its restricted key in the
  CI platform's encrypted secret store and expose it only to that job.
- **Production:** store the key in the hosting platform's secret manager and
  inject it at runtime. Do not bake it into an image, commit it, pass it in a
  URL, or place it in deployment documentation.

The Postgres username and password defaults in `docker-compose.yml` are for the
isolated local Compose network. A deployment must replace them with generated
credentials from its database or secret manager.

### Key lifecycle

1. **Create:** create a dedicated key for one environment. Apply the narrowest
   permissions, budget, and usage limits the provider supports.
2. **Store:** save the key only in the environment's approved secret store.
   Keep placeholders, names, and setup instructions in the repository.
3. **Use:** inject the key into the process and unwrap it only at the provider
   client boundary.
4. **Rotate:** create a replacement key, update the secret store, redeploy, and
   verify one provider-backed request before revoking the old key.
5. **Revoke:** disable keys that are unused, superseded, or suspected to be
   exposed. Review provider usage and application logs for unexpected access.

### Suspected exposure

Revoke the exposed key immediately. Create a replacement, update the affected
secret stores, redeploy, and verify a provider-backed request. Review provider
usage for unexpected cost or models, confirm logs contain redaction markers
rather than secret values, and record the incident without copying the key into
the report.

## Reporting a vulnerability

Do not open a public issue containing credentials, receipt data, or exploit
details. Contact the repository owner privately with reproduction steps and the
affected version or commit. Revoke any credential included in a report.
