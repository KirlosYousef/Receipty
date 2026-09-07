class ProviderError(Exception):
    """Upstream model provider failed after retries or returned a hard error."""


class CreditsExhausted(ProviderError):
    """Provider rejected the request for billing / credit reasons (HTTP 402)."""


class DailyLimitReached(ProviderError):
    """Free-tier daily request cap hit; credits usually do not lift this."""
