"""Asynchronous Webhook Dispatcher & Delivery Engine for ReturnShield AI.

Features:
- Persistent subscription management in storage/webhooks.json.
- Canonical-to-platform payload adaptation (Shopify, Narvar, Amazon SAFE-T, Flipkart).
- Cryptographic HMAC-SHA256 signing (hex and base64).
- Asynchronous dispatch via httpx with exponential backoff retry on 5xx/connection drops.
- Persistent delivery audit logging in storage/webhook_deliveries.json.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union
import uuid

import httpx

from core.erp_adapters import adapt_canonical_payload
from core.erp_schemas import (
    CanonicalWebhookPayload,
    ERPPlatform,
    WebhookDeliveryAttempt,
    WebhookEvent,
    WebhookSubscription,
)
from core.webhook_security import build_outbound_webhook_headers

logger = logging.getLogger("returnshield.webhook_dispatcher")

DEFAULT_STORAGE_DIR = Path(__file__).resolve().parent.parent / "storage"


class WebhookDispatcher:
    """Manages ERP webhook subscriptions and dispatches tamper-proof signed payloads."""

    def __init__(
        self,
        storage_dir: Optional[Union[str, Path]] = None,
        max_retries: int = 3,
        initial_backoff_sec: float = 0.2,
        request_timeout_sec: float = 5.0,
    ):
        self.storage_dir = Path(storage_dir or DEFAULT_STORAGE_DIR)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.subscriptions_file = self.storage_dir / "webhooks.json"
        self.deliveries_file = self.storage_dir / "webhook_deliveries.json"

        self.max_retries = max_retries
        self.initial_backoff_sec = initial_backoff_sec
        self.request_timeout_sec = request_timeout_sec

        self._subscriptions: Dict[str, WebhookSubscription] = {}
        self._deliveries: List[WebhookDeliveryAttempt] = []

        self._load_state()

    def _load_state(self) -> None:
        """Loads subscriptions and deliveries from disk."""
        if self.subscriptions_file.exists():
            try:
                with open(self.subscriptions_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data:
                        sub = WebhookSubscription(**item)
                        self._subscriptions[sub.subscription_id] = sub
                logger.info("Loaded %d webhook subscriptions from disk.", len(self._subscriptions))
            except Exception as e:
                logger.warning("Error loading webhook subscriptions: %s", e)

        if self.deliveries_file.exists():
            try:
                with open(self.deliveries_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._deliveries = [WebhookDeliveryAttempt(**item) for item in data[-200:]]
            except Exception as e:
                logger.warning("Error loading webhook deliveries: %s", e)

    def _persist_subscriptions(self) -> None:
        """Saves active subscriptions to disk."""
        try:
            with open(self.subscriptions_file, "w", encoding="utf-8") as f:
                json.dump([s.model_dump(mode="json") for s in self._subscriptions.values()], f, indent=2)
        except Exception as e:
            logger.error("Failed persisting webhook subscriptions: %s", e)

    def _persist_deliveries(self) -> None:
        """Saves last 200 delivery audit records to disk."""
        try:
            with open(self.deliveries_file, "w", encoding="utf-8") as f:
                json.dump([d.model_dump(mode="json") for d in self._deliveries[-200:]], f, indent=2)
        except Exception as e:
            logger.error("Failed persisting webhook delivery log: %s", e)

    # ==========================================================================
    # Subscription Management
    # ==========================================================================

    def register_subscription(
        self,
        target_url: str,
        secret_token: str,
        platform: ERPPlatform = ERPPlatform.CUSTOM,
        subscribed_events: Optional[List[WebhookEvent]] = None,
        description: Optional[str] = None,
    ) -> WebhookSubscription:
        """Registers a new webhook listener."""
        events = subscribed_events or [
            WebhookEvent.RETURN_INSPECTED,
            WebhookEvent.REFUND_APPROVED,
            WebhookEvent.REFUND_PARTIAL,
            WebhookEvent.REFUND_REJECTED,
            WebhookEvent.FRAUD_ALERT,
        ]
        sub = WebhookSubscription(
            platform=platform,
            target_url=target_url,
            secret_token=secret_token,
            subscribed_events=events,
            description=description,
        )
        self._subscriptions[sub.subscription_id] = sub
        self._persist_subscriptions()
        logger.info("Registered webhook subscription %s for platform %s -> %s",
                    sub.subscription_id, platform.value, target_url)
        return sub

    def get_subscription(self, subscription_id: str) -> Optional[WebhookSubscription]:
        return self._subscriptions.get(subscription_id)

    def list_subscriptions(self, active_only: bool = True) -> List[WebhookSubscription]:
        subs = list(self._subscriptions.values())
        if active_only:
            subs = [s for s in subs if s.is_active]
        return subs

    def delete_subscription(self, subscription_id: str) -> bool:
        if subscription_id in self._subscriptions:
            del self._subscriptions[subscription_id]
            self._persist_subscriptions()
            logger.info("Deleted webhook subscription %s", subscription_id)
            return True
        return False

    def list_deliveries(self, limit: int = 50) -> List[WebhookDeliveryAttempt]:
        return list(reversed(self._deliveries[-limit:]))

    # ==========================================================================
    # Outbound Dispatch Engine
    # ==========================================================================

    async def deliver_to_subscription(
        self,
        subscription: WebhookSubscription,
        canonical_payload: CanonicalWebhookPayload,
        client: Optional[httpx.AsyncClient] = None,
    ) -> WebhookDeliveryAttempt:
        """Delivers a signed payload to a single subscription endpoint with exponential retry."""
        # 1. Adapt canonical payload into target platform structure
        adapted = adapt_canonical_payload(canonical_payload, subscription.platform)
        
        # Populate platform_payload in canonical for audit
        payload_to_send = canonical_payload.model_copy()
        payload_to_send.platform_payload = adapted
        
        payload_json = payload_to_send.model_dump_json()
        payload_bytes = payload_json.encode("utf-8")

        # 2. Build secure cryptographic HMAC headers
        headers = build_outbound_webhook_headers(
            payload_bytes=payload_bytes,
            secret=subscription.secret_token,
            event_topic=canonical_payload.event.value,
            subscription_id=subscription.subscription_id,
            platform=subscription.platform.value,
        )

        should_close_client = False
        if client is None:
            client = httpx.AsyncClient(timeout=self.request_timeout_sec)
            should_close_client = True

        status_code = None
        success = False
        error_msg = None
        resp_text = None
        attempts = 0
        t0 = time.perf_counter()

        try:
            for attempt in range(1, self.max_retries + 1):
                attempts = attempt
                try:
                    res = await client.post(
                        subscription.target_url,
                        content=payload_bytes,
                        headers=headers,
                    )
                    status_code = res.status_code
                    resp_text = res.text[:500] if res.text else None

                    # Success on 2xx status codes
                    if 200 <= status_code < 300:
                        success = True
                        break
                    elif status_code in [400, 401, 403, 404, 422]:
                        # Non-retryable client error
                        error_msg = f"HTTP {status_code} client error from target"
                        break
                    else:
                        error_msg = f"HTTP {status_code} server response"

                except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError) as exc:
                    error_msg = f"{type(exc).__name__}: {str(exc)}"

                # Exponential backoff before retry
                if attempt < self.max_retries:
                    backoff = self.initial_backoff_sec * (2 ** (attempt - 1))
                    await asyncio.sleep(backoff)

        finally:
            if should_close_client:
                await client.aclose()

        duration_ms = (time.perf_counter() - t0) * 1000.0

        record = WebhookDeliveryAttempt(
            subscription_id=subscription.subscription_id,
            target_url=subscription.target_url,
            event=canonical_payload.event,
            success=success,
            status_code=status_code,
            duration_ms=round(duration_ms, 2),
            attempts=attempts,
            request_headers=headers,
            response_body=resp_text,
            error_message=error_msg,
        )

        self._deliveries.append(record)
        self._persist_deliveries()

        logger.info(
            "Webhook delivery [%s] to %s: status=%s, success=%s, attempts=%d, duration=%.1fms",
            canonical_payload.event.value,
            subscription.target_url,
            status_code,
            success,
            attempts,
            duration_ms,
        )
        return record

    async def dispatch_event(
        self,
        canonical_payload: CanonicalWebhookPayload,
    ) -> List[WebhookDeliveryAttempt]:
        """Dispatches an event to all matching active subscribers asynchronously."""
        active_subs = [
            s for s in self._subscriptions.values()
            if s.is_active and canonical_payload.event in s.subscribed_events
        ]

        if not active_subs:
            logger.info("No active webhook subscriptions match event '%s'.", canonical_payload.event.value)
            return []

        async with httpx.AsyncClient(timeout=self.request_timeout_sec) as client:
            tasks = [
                self.deliver_to_subscription(sub, canonical_payload, client=client)
                for sub in active_subs
            ]
            deliveries = await asyncio.gather(*tasks, return_exceptions=False)

        return list(deliveries)

