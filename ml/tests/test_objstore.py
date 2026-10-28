from ml import objstore


def test_endpoint_url_forms():
    assert objstore.endpoint_url("localhost:9000", False) == ("http://localhost:9000", True)
    assert objstore.endpoint_url("rustfs:9000", True) == ("https://rustfs:9000", True)
    assert objstore.endpoint_url("https://t3.storageapi.dev", False) == (
        "https://t3.storageapi.dev", False)


def test_config_from_env(monkeypatch):
    monkeypatch.delenv("OBJECT_STORE_ENDPOINT", raising=False)
    monkeypatch.delenv("OBJECT_STORE_PATH_STYLE", raising=False)
    cfg = objstore.config_from_env()
    assert cfg.endpoint == "http://localhost:9000" and cfg.path_style
    assert cfg.region == "us-east-1"

    monkeypatch.setenv("OBJECT_STORE_ENDPOINT", "https://bucket.example")
    monkeypatch.setenv("OBJECT_STORE_REGION", "auto")
    cfg = objstore.config_from_env()
    assert cfg.endpoint == "https://bucket.example" and not cfg.path_style
    assert cfg.region == "auto"
    monkeypatch.setenv("OBJECT_STORE_PATH_STYLE", "true")
    assert objstore.config_from_env().path_style
