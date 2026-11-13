"""Decides, for every guarded request, who is asking and whether their role reaches that area."""

from typing import Annotated, Optional

from fastapi import Header, HTTPException, Request

from config import config
from persistence.database import session_factory
from persistence.models import UserAccount
from persistence.repository import fetch_user_account


def authorize_request(request: Request,
                      session_token: Annotated[Optional[str],
                                               Header(alias=config.SESSION_TOKEN_HEADER)] = None):
    """The signed-in account, or 401 / 403 before the endpoint runs."""
    if not session_token:
        raise HTTPException(status_code=401, detail=config.SIGN_IN_REQUIRED_MESSAGE)
    with session_factory() as session:
        account = fetch_user_account(session, UserAccount.session_token == session_token)
    if account is None:
        raise HTTPException(status_code=401, detail=config.SIGN_IN_REQUIRED_MESSAGE)
    route = request.scope.get("route")
    path = getattr(route, "path", request.url.path)
    area = config.AREA_BY_ROUTE.get((request.method, path))
    if area not in config.ROLE_AREA_ACCESS.get(account["role"], []):
        raise HTTPException(status_code=403, detail=config.AREA_REFUSED_MESSAGE.format(
            role=account["role"], area=area))
    return account
