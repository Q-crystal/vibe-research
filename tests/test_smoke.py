import vibe_research as vr


def test_version():
    assert vr.version == "0.1.0"


def test_import():
    from vibe_research import cli

    assert hasattr(cli, "main")
