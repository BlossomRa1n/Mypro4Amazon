"""Identity-only parser must never reach event/label payloads."""
import ast
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import pickle
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("extract_pickle_identity", ROOT / "tools/extract_pickle_identity.py")
identity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(identity)


def forbidden_payload():
    raise AssertionError("Event data must not be unpickled")


class Payload:
    def __reduce__(self):
        return forbidden_payload, ()


def fixture(module_name="future_window_data", class_name="FutureWindowData", protocol=4,
            wrapper="bare"):
    module = types.ModuleType(module_name)
    cls = type(class_name, (), {"__module__": module_name})
    setattr(module, class_name, cls)
    obj = cls()
    obj.hist_len, obj.seed = 50, 42
    # More than one pickle APPENDS batch; repeated values across lists use memo references.
    obj.users = ["__PAD__", "__UNKNOWN__"] + ["u%05d" % x for x in range(2500)]
    obj.items = ["__PAD__", "__UNKNOWN__", "item_a", "item_b"]
    obj.categories = ["__PAD__", "__UNKNOWN__", "中文"]
    obj.brands = ["__PAD__", "__UNKNOWN__", "brand"]
    obj.uid = Payload()
    obj.iid = Payload()
    obj.labels = Payload()
    with patch.dict(sys.modules, {module_name: module}):
        cached = {"bare": obj, "baseline": {"prep_id": "fixture-prep-id", "data": obj},
                  "smoke": {"data": obj}, "wrong": {"other": obj}}[wrapper]
        return pickle.dumps(cached, protocol=protocol), obj


class IdentityTests(unittest.TestCase):
    def test_protocols_and_original_fingerprint(self):
        # Execute the actual tiny function without importing pandas/torch.
        tree = ast.parse((ROOT / "code/baseline_data.py").read_text())
        fn = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "fingerprint")
        namespace = {"hashlib": hashlib, "json": json}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "baseline-fingerprint", "exec"), namespace)
        for module, cls in identity.CLASSES:
            for protocol in (2, 3, 4, 5):
                with self.subTest(module=module, protocol=protocol):
                    blob, obj = fixture(module, cls, protocol)
                    expected = namespace["fingerprint"]([obj.users, obj.items, obj.brands, obj.categories])
                    # Removing every byte after the uid key still succeeds.
                    values, boundary = identity.parse_prefix(io.BytesIO(blob))
                    shortened, _ = identity.parse_prefix(io.BytesIO(blob[:boundary]))
                    self.assertEqual(values, shortened)
                    self.assertLess(boundary, len(blob))
                    with tempfile.TemporaryDirectory() as tmp:
                        path = Path(tmp) / "data.pkl"
                        path.write_bytes(blob)
                        result = identity.extract(path, hashlib.sha256(blob).hexdigest(), expected)
                        self.assertEqual(result["users"], obj.users)
                        self.assertEqual(result["users"][2501], "u02499")
                        self.assertEqual(result["encoders_hash"], expected)
                        with self.assertRaises(identity.IdentityError):
                            identity.extract(path, expected_sha256="0" * 64)
                        with self.assertRaises(identity.IdentityError):
                            identity.extract(path, expected_fingerprint="0" * 64)

    def test_wrong_missing_truncated_and_oversized_sources(self):
        with self.assertRaises(FileNotFoundError):
            identity.extract("/does-not-exist/identity.pkl")
        for blob in (pickle.dumps({"users": ["u"]}), fixture("wrong_module", "WrongClass")[0],
                     fixture()[0][:100], pickle.dumps(Payload()), fixture(wrapper="wrong")[0]):
            with self.subTest(size=len(blob)), self.assertRaises(ValueError):
                identity.parse_prefix(io.BytesIO(blob))
        with self.assertRaises(identity.IdentityError):
            identity.parse_prefix(io.BytesIO(fixture()[0]), max_prefix_bytes=100)

    def test_real_baseline_cache_and_smoke_wrappers(self):
        for module, cls in identity.CLASSES:
            for wrapper in ("baseline", "smoke"):
                for protocol in (2, 3, 4, 5):
                    with self.subTest(module=module, wrapper=wrapper, protocol=protocol):
                        blob, obj = fixture(module, cls, protocol, wrapper)
                        encoders, boundary = identity.parse_prefix(io.BytesIO(blob))
                        self.assertEqual(encoders["users"], obj.users)
                        self.assertEqual(identity.fingerprint([encoders[x] for x in identity.FIELDS]),
                                         identity.fingerprint([obj.users, obj.items, obj.brands, obj.categories]))
                        truncated, _ = identity.parse_prefix(io.BytesIO(blob[:boundary]))
                        self.assertEqual(truncated, encoders)


if __name__ == "__main__":
    unittest.main()
