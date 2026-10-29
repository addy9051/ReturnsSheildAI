"""Cryptographic Signature & Replay Defense for ReturnShield Webhooks.

Implements official security standards:
- Shopify Webhook Verification (HMAC-SHA256 Base64 via X-Shopify-Hmac-SHA256).
- Standard / Stripe / GitHub Webhook Signatures (HMAC-SHA256 Hex via X-ReturnShield-Signature).
- Timing-safe comparison (hmac.compare_digest) to thwart timing side-channels.
- Timestamp freshness & tolerance check (default 300s window) against replay attacks.
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import hmac
import logging
import time
from typing import Dict, Optional, Tuple, Union

logger = logging.getLogger("returnshield.webhook_security")

DEFAULT_TOLERANCE_SECONDS = 300  # 5-minute replay window


def sign_payload_hex(payload_bytes: bytes, secret: str) -> str:
    """Computes standard hex-encoded HMAC-SHA256 signature for payload."""
    mac = hmac.new(secret.encode("utf-8"), payload_bytes, hashlib.sha256)
    return mac.hexdigest()


def sign_payload_base64(payload_bytes: bytes, secret: str) -> str:
    """Computes official Shopify-standard Base64-encoded HMAC-SHA256 signature."""
    mac = hmac.new(secret.encode("utf-8"), payload_bytes, hashlib.sha256)
    return base64.b64encode(mac.digest()).decode("utf-8")


def verify_signature_hex(
    payload_bytes: bytes,
    secret: str,
    signature_header: str,
) -> bool:
    """Verifies standard hex-encoded HMAC-SHA256 signature using constant-time comparison.

    Accepts bare hex or prefixed 'sha256=<hex>'.
    """
    if not signature_header or not secret:
        return False

    clean_sig = signature_header.strip()
    if clean_sig.startswith("sha256="):
        clean_sig = clean_sig[7:]

    expected = sign_payload_hex(payload_bytes, secret)
    return hmac.compare_digest(expected, clean_sig)


def verify_signature_base64(
    payload_bytes: bytes,
    secret: str,
    signature_header: str,
) -> bool:
    """Verifies official Shopify Base64-encoded HMAC-SHA256 signature."""
    if not signature_header or not secret:
        return False

    clean_sig = signature_header.strip()
    expected = sign_payload_base64(payload_bytes, secret)
    return hmac.compare_digest(expected, clean_sig)


def verify_timestamp_freshness(
    timestamp_input: Union[int, float, str],
    tolerance_seconds: int = DEFAULT_TOLERANCE_SECONDS,
) -> bool:
    """Ensures incoming webhook timestamp is within tolerance window (default 300s).

    Supports Unix epoch integers/floats or ISO 8601 strings.
    """
    current_epoch = time.time()

    if isinstance(timestamp_input, (int, float)):
        ts_epoch = float(timestamp_input)
    elif isinstance(timestamp_input, str):
        try:
            # Check if numeric string
            ts_epoch = float(timestamp_input)
        except ValueError:
            # Parse ISO-8601 string
            try:
                dt = datetime.fromisoformat(timestamp_input.replace("Z", "+00:00"))
                ts_epoch = dt.timestamp()
            except Exception:
                logger.warning("Unparseable webhook timestamp format: %s", timestamp_input)
                return False
    else:
        return False

    diff = abs(current_epoch - ts_epoch)
    if diff > tolerance_seconds:
        logger.warning(
            "Webhook timestamp rejected: drift of %.1fs exceeds tolerance of %ds",
            diff,
            tolerance_seconds,
        )
        return False
    return True


def build_outbound_webhook_headers(
    payload_bytes: bytes,
    secret: str,
    event_topic: str,
    subscription_id: str,
    platform: str = "custom",
) -> Dict[str, str]:
    """Constructs comprehensive HTTP headers for outbound webhook transmission."""
    now_epoch = int(time.time())
    iso_now = datetime.now(timezone.utc).isoformat()

    hex_sig = sign_payload_hex(payload_bytes, secret)
    b64_sig = sign_payload_base64(payload_bytes, secret)

    headers = {
        "Content-Type": "application/json",
        "User-Agent": "ReturnShield-AI-Webhook-Engine/1.0",
        "X-ReturnShield-Signature": f"sha256={hex_sig}",
        "X-ReturnShield-Timestamp": str(now_epoch),
        "X-ReturnShield-Event": event_topic,
        "X-ReturnShield-Subscription-Id": subscription_id,
        "X-ReturnShield-Delivered-At": iso_now,
    }

    # Add platform-tailored header aliases
    if platform.lower() == "shopify":
        headers["X-Shopify-Hmac-SHA256"] = b64_sig
        headers["X-Shopify-Topic"] = event_topic
    elif platform.lower() == "narvar":
        headers["X-Narvar-Topic"] = event_topic
        headers["X-Narvar-Timestamp"] = str(now_epoch)

    return headers

