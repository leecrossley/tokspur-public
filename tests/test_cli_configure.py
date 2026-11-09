# SPDX-License-Identifier: Apache-2.0
"""Configuration check that runs on a clean checkout and in CI, with no model files."""


def test_configuration_links_raw_gguf_not_prunable_generated_folder(
    tmp_path, monkeypatch
):

    from tokspur import cli, local

    policy = local.profile()
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(local, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "profile", lambda: policy)
    monkeypatch.setattr(cli, "port_available", lambda port: True)
    monkeypatch.setattr(cli, "check_adoption", lambda data: None)
    target, draft = tmp_path / "existing", tmp_path / "draft"
    target.mkdir()
    draft.mkdir()
    part = target / "fixture.gguf"
    part.write_bytes(b"not real weights")
    data = {
        "target": {"directory": str(target), "assets": [{"path": str(part)}]},
        "draft": {"directory": str(draft)},
    }
    local.write_json(tmp_path / ".tokspur/adoption.json", data)
    cli.configure(18081)
    state = tmp_path / ".tokspur/state"
    assert (state / "models/tokspur").resolve() == target
    assert not (state / "gguf").exists()
    configured = local.read_json(tmp_path / ".tokspur/configured.json")
    assert configured["port"] == 18081
    assert configured["model"] == "tokspur"
    assert part.read_bytes() == b"not real weights"
