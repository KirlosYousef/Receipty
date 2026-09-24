# Security

## Provider API keys

Receipty reads `OPENROUTER_API_KEY` at runtime. Pydantic stores it as a
redacted secret value and the application unwraps it only when constructing an
OpenRouter client. Do not print, serialize, trace, or return the unwrapped
value.

Treat configuration and secrets differently. A model name or timeout can be
committed because it describes behavior. An API key grants access to an
external account and must stay outside the repository.

### Storage by environment

- **Local development:** copy `.env.example` to the gitignored `.env` file and
  put the key there. Docker Compose reads only the explicitly listed variables.
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
