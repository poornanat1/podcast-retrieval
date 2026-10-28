import json

import pytest

from ml.serving import bundle as bundle_mod
from ml.serving import bundlestore


class FakeS3:
    """The subset of the boto3 S3 client the bundle store uses."""

    def __init__(self):
        self.objects: dict[str, bytes] = {}

        class ClientError(Exception):
            def __init__(self, code):
                self.response = {"Error": {"Code": code}}

        self.exceptions = type("E", (), {"ClientError": ClientError})
        self._error = ClientError

    def head_object(self, Bucket, Key):
        if Key not in self.objects:
            raise self._error("404")

    def upload_file(self, path, Bucket, Key, ExtraArgs=None):
        self.objects[Key] = open(path, "rb").read()

    def get_object(self, Bucket, Key):
        import io
        return {"Body": io.BytesIO(self.objects[Key])}

    def download_file(self, Bucket, Key, path):
        open(path, "wb").write(self.objects[Key])


def make_bundle(root, files):
    bundle = root / "bundle"
    bundle.mkdir()
    hashes = {}
    for name, body in files.items():
        (bundle / name).write_bytes(body)
        hashes[name] = bundle_mod.sha256(bundle / name)
    manifest = {"bundle_id": bundle_mod.bundle_id(hashes), "files": hashes}
    (bundle / bundle_mod.MANIFEST).write_text(json.dumps(manifest))
    return bundle, manifest["bundle_id"]


def test_publish_then_fetch_round_trip(tmp_path, monkeypatch):
    fake = FakeS3()
    monkeypatch.setattr(bundlestore, "client_from_env", lambda var, default: (fake, "models"))
    bundle, bundle_id = make_bundle(tmp_path, {"a.onnx": b"aaa", "ranker.json": b"{}"})

    assert bundlestore.publish(bundle) == f"models/bundles/{bundle_id}"
    assert bundlestore.publish(bundle).endswith("(already published)")
    assert sorted(fake.objects) == [f"bundles/{bundle_id}/a.onnx",
                                    f"bundles/{bundle_id}/manifest.json",
                                    f"bundles/{bundle_id}/ranker.json"]

    dest = tmp_path / "cache"
    got = bundlestore.fetch(bundle_id, dest)
    assert got == dest / bundle_id
    assert (got / "a.onnx").read_bytes() == b"aaa"
    # A second fetch is served from the cache without touching the store.
    fake.objects.clear()
    assert bundlestore.fetch(bundle_id, dest) == got


def test_fetch_rejects_tampered_objects(tmp_path, monkeypatch):
    fake = FakeS3()
    monkeypatch.setattr(bundlestore, "client_from_env", lambda var, default: (fake, "models"))
    bundle, bundle_id = make_bundle(tmp_path, {"a.onnx": b"aaa"})
    bundlestore.publish(bundle)
    fake.objects[f"bundles/{bundle_id}/a.onnx"] = b"bbb"
    with pytest.raises(ValueError):
        bundlestore.fetch(bundle_id, tmp_path / "cache")
    assert not (tmp_path / "cache" / bundle_id).exists()
