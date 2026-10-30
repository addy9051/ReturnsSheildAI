"""Authentication, Authorization, and Multi-Tenant Identity Layer for ReturnShield AI.

Provides role-based access control (RBAC) and tenant isolation for:
- Catalog and webhook administration (ADMIN role)
- Return inspection scans and pickup authorization (INSPECTOR role)
- Certificate retrieval and dispute audit verification (AUDITOR / READ-ONLY role)
"""

from __future__ import annotations

from enum import Enum
import logging
import os
from typing import Optional

from fastapi import Header, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

logger = logging.getLogger("returnshield.auth")

# Default service keys (can be overridden via environment variables)
DEFAULT_ADMIN_KEY = os.getenv("RETURNSHIELD_ADMIN_KEY", "rs_admin_key_sec_2026")
DEFAULT_INSPECTOR_KEY = os.getenv("RETURNSHIELD_INSPECTOR_KEY", "rs_inspector_key_sec_2026")
DEFAULT_AUDITOR_KEY = os.getenv("RETURNSHIELD_AUDITOR_KEY", "rs_auditor_key_sec_2026")
SIGNING_SECRET = os.getenv("RETURNSHIELD_SIGNING_SECRET", "rs_hmac_signing_secret_dev_2026")

# Environment flag to enforce auth (defaults to True)
AUTH_ENFORCED = os.getenv("RETURNSHIELD_AUTH_ENFORCED", "true").lower() in ("true", "1", "yes")


class ServiceRole(str, Enum):
    """Hierarchical service role permissions."""
    ADMIN = "ADMIN"
    INSPECTOR = "INSPECTOR"
    AUDITOR = "AUDITOR"


ROLE_HIERARCHY = {
    ServiceRole.ADMIN: 3,
    ServiceRole.INSPECTOR: 2,
    ServiceRole.AUDITOR: 1,
}


class AuthenticatedIdentity(BaseModel):
    """Represents verified caller identity and tenant boundary."""
    role: ServiceRole
    tenant_id: str
    identity_name: str


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def extract_api_key(request: Request, x_api_key: Optional[str] = None) -> Optional[str]:
    """Extracts API key from X-API-Key or Authorization Bearer header."""
    if x_api_key:
        return x_api_key.strip()
    
    auth_header = request.headers.get("Authorization")
    if auth_header:
        parts = auth_header.strip().split()
        if len(parts) == 2 and parts[0].lower() in ("bearer", "apikey"):
            return parts[1]
        elif len(parts) == 1:
            return parts[0]
    return None


async def get_authenticated_identity(
    request: Request,
    x_api_key: Optional[str] = Security(api_key_header),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
) -> AuthenticatedIdentity:
    """Verifies service credentials and resolves tenant boundary."""
    tenant_id = (x_tenant_id or "tenant-default").strip()
    if not tenant_id:
        tenant_id = "tenant-default"

    raw_key = extract_api_key(request, x_api_key)

    if not raw_key:
        if not AUTH_ENFORCED:
            # Dev-mode bypass
            return AuthenticatedIdentity(
                role=ServiceRole.ADMIN,
                tenant_id=tenant_id,
                identity_name="dev-unauthenticated-bypass",
            )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing required authentication credentials. Provide 'X-API-Key' or 'Authorization: Bearer <key>'.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    admin_key = os.getenv("RETURNSHIELD_ADMIN_KEY", DEFAULT_ADMIN_KEY)
    inspector_key = os.getenv("RETURNSHIELD_INSPECTOR_KEY", DEFAULT_INSPECTOR_KEY)
    auditor_key = os.getenv("RETURNSHIELD_AUDITOR_KEY", DEFAULT_AUDITOR_KEY)

    # Constant-time comparison
    import hmac

    if hmac.compare_digest(raw_key, admin_key):
        return AuthenticatedIdentity(
            role=ServiceRole.ADMIN,
            tenant_id=tenant_id,
            identity_name="admin-service",
        )
    elif hmac.compare_digest(raw_key, inspector_key):
        return AuthenticatedIdentity(
            role=ServiceRole.INSPECTOR,
            tenant_id=tenant_id,
            identity_name="inspector-service",
        )
    elif hmac.compare_digest(raw_key, auditor_key):
        return AuthenticatedIdentity(
            role=ServiceRole.AUDITOR,
            tenant_id=tenant_id,
            identity_name="auditor-service",
        )
    else:
        logger.warning("Authentication failed: invalid service key provided from tenant '%s'", tenant_id)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid service API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )


def require_role(min_role: ServiceRole):
    """FastAPI dependency ensuring caller possesses at least the specified role."""
    async def role_checker(
        identity: AuthenticatedIdentity = Security(get_authenticated_identity),
    ) -> AuthenticatedIdentity:
        caller_weight = ROLE_HIERARCHY.get(identity.role, 0)
        required_weight = ROLE_HIERARCHY.get(min_role, 0)

        if caller_weight < required_weight:
            logger.warning(
                "Access forbidden: role '%s' lacks required permission '%s' for tenant '%s'",
                identity.role.value, min_role.value, identity.tenant_id
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Forbidden: role '{identity.role.value}' lacks required '{min_role.value}' permission.",
            )
        return identity

    return role_checker


# Pre-defined dependencies for endpoint decoration
require_admin = require_role(ServiceRole.ADMIN)
require_inspector = require_role(ServiceRole.INSPECTOR)
require_auditor = require_role(ServiceRole.AUDITOR)

