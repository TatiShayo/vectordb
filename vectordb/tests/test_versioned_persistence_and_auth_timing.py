import os, sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from api.auth import _matches_any, _resolve, is_admin
from config import ADMIN_KEYS, USER_KEYS, ALL_KEYS
from fastapi import HTTPException

def test_matches_any_valid_keys():
    for key in ALL_KEYS:
        assert _matches_any(key, ALL_KEYS) is True

def test_matches_any_invalid_keys():
    assert _matches_any('invalid-key-xyz', ALL_KEYS) is False
    assert _matches_any('', ALL_KEYS) is False
    assert _matches_any('user-sec', ALL_KEYS) is False

def test_admin_privilege_scoping():
    for adm in ADMIN_KEYS:
        assert is_admin(adm) is True
    for usr in USER_KEYS:
        if usr not in ADMIN_KEYS:
            assert is_admin(usr) is False

def test_resolve_rejection():
    with pytest.raises(HTTPException) as exc1:
        _resolve(None)
    assert exc1.value.status_code == 401

    with pytest.raises(HTTPException) as exc2:
        _resolve('completely-bogus-token')
    assert exc2.value.status_code == 403
