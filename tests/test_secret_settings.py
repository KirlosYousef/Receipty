from app.core.config import Settings


def test_secret_file_is_loaded_without_exposing_value_in_settings_output(
    monkeypatch, tmp_path
):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    (tmp_path / "openrouter_api_key").write_text("test-secret-from-file")

    settings = Settings(_env_file=None, _secrets_dir=tmp_path)

    assert settings.openrouter_api_key.get_secret_value() == "test-secret-from-file"
    assert "test-secret-from-file" not in repr(settings)
    assert "test-secret-from-file" not in settings.model_dump_json()
