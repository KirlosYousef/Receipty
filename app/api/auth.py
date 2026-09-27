import base64
import binascii
import hmac

from fastapi import Request

from app.core.config import Settings


def owner_authorized(request: Request, settings: Settings) -> bool:
    """Check one demo owner's HTTP Basic credentials without logging them."""
    header = request.headers.get("Authorization", "")
    scheme, _, encoded = header.partition(" ")
    if scheme.lower() != "basic" or not encoded:
        return False
    try:
        raw = base64.b64decode(encoded, validate=True)
    except binascii.Error:
        return False
    username, separator, password = raw.partition(b":")
    if not separator:
        return False
    return hmac.compare_digest(
        username, settings.auth_username.encode("utf-8")
    ) and hmac.compare_digest(
        password, settings.auth_password.get_secret_value().encode("utf-8")
    )
