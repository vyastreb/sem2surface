import pytest

from sem2surface_cli import build_parser, main


def test_cli_exposes_version(capsys):
    parser = build_parser()
    with pytest.raises(SystemExit) as raised:
        parser.parse_args(["--version"])
    assert raised.value.code == 0
    assert "0.2.2" in capsys.readouterr().out


def test_cli_requires_three_images(capsys):
    with pytest.raises(SystemExit) as raised:
        main(["one.tif", "two.tif"])
    assert raised.value.code == 2
    assert "at least three" in capsys.readouterr().err
