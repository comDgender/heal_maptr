"""Tests for heal.models.base_modules __init__."""


def test_base_modules_public_exports():
    from heal.models.base_modules import (
        BiFPN,
        DwUnet,
        SECONDNeck,
        get_fpn_config,
    )

    assert BiFPN is not None
    assert DwUnet is not None
    assert SECONDNeck is not None
    assert callable(get_fpn_config)
