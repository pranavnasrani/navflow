"""Tests for app.tunnel: the ngrok command line, reading its address, and a missing install."""
import app.tunnel as tunnel
from app.tunnel import NgrokTunnel, ngrok_command, parse_public_url


def test_command_forwards_the_remote_port():
    cmd = ngrok_command(8765)
    assert cmd[:3] == ["ngrok", "http", "8765"] and not any(c.startswith("--url") for c in cmd)


def test_command_with_a_reserved_domain():
    for given in ("navflow.ngrok-free.app", "https://navflow.ngrok-free.app/"):
        assert "--url=navflow.ngrok-free.app" in ngrok_command(8765, given)


def test_parse_public_url_prefers_https():
    reply = {"tunnels": [{"public_url": "http://a.ngrok-free.app"}, {"public_url": "https://a.ngrok-free.app"}]}
    assert parse_public_url(reply) == "https://a.ngrok-free.app"
    assert parse_public_url({"tunnels": []}) is None and parse_public_url({}) is None


def test_missing_ngrok_says_how_to_install(monkeypatch):
    said = []
    monkeypatch.setattr(tunnel.shutil, "which", lambda name: None)
    assert NgrokTunnel(8765, say=said.append).start() is False
    assert "brew install ngrok" in said[0] and "add-authtoken" in said[0]


def test_token_error_is_explained():
    t = NgrokTunnel(8765)
    t._tail = ['{"err":"authentication failed: ERR_NGROK_4018"}']
    assert "add-authtoken" in t._reason()
