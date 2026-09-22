#!/usr/bin/env python3
"""Read only early encoder lists in project data pickles, without unpickling.

Standard library only. Stops at the `uid` state key, before any event arrays.
SHA256 reads the complete source as opaque chunks; no later opcodes are parsed.
The JSON users array preserves encoded offsets 0/1 (PAD/UNKNOWN).
Accepts a bare data object or the project's {'prep_id': str, 'data': object}
and smoke-test {'data': object} cache wrappers.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickletools
import tempfile

FIELDS = ("users", "items", "brands", "categories")
CLASSES = {("baseline_data", "BenchmarkData"), ("future_window_data", "FutureWindowData")}
MARK = object()
ROOT = object()
STATE = object()
WRAPPER = object()


class IdentityError(ValueError):
    pass


class BoundedReader:
    def __init__(self, stream, max_prefix_bytes):
        self.stream = stream
        self.limit = max_prefix_bytes

    def read(self, n=-1):
        if n < 0 or n > 1024 * 1024 or self.stream.tell() + n > self.limit:
            raise IdentityError("Identity prefix exceeds parser limits")
        return self.stream.read(n)

    def readline(self):
        value = self.stream.readline(1024 * 1024 + 1)
        if len(value) > 1024 * 1024 or self.stream.tell() > self.limit:
            raise IdentityError("Identity prefix exceeds parser limits")
        return value

    def tell(self):
        return self.stream.tell()


def fingerprint(value):
    digest = hashlib.sha256()
    # Same bytes as baseline_data.fingerprint, without a second full JSON copy.
    for chunk in json.JSONEncoder(sort_keys=True, ensure_ascii=True).iterencode(value):
        digest.update(chunk.encode())
    return digest.hexdigest()


def parse_prefix(stream, max_prefix_bytes=128 * 1024 * 1024):
    stack, memo = [], {}
    root_seen = False
    root_index = 0
    try:
        for opcode, arg, _ in pickletools.genops(BoundedReader(stream, max_prefix_bytes)):
            op = opcode.name
            if op in {"PROTO", "FRAME"}:
                continue
            if op in {"SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "UNICODE",
                      "BININT", "BININT1", "BININT2", "INT", "LONG", "LONG1", "LONG4"}:
                stack.append(arg)
            elif op == "GLOBAL":
                pair = tuple(arg.split(" "))
                if pair not in CLASSES:
                    raise IdentityError("Unsupported source class")
                stack.append(pair)
            elif op == "STACK_GLOBAL":
                name, module = stack.pop(), stack.pop()
                if (module, name) not in CLASSES:
                    raise IdentityError("Unsupported source class")
                stack.append((module, name))
            elif op == "EMPTY_TUPLE":
                stack.append(())
            elif op == "NEWOBJ":
                args, cls = stack.pop(), stack.pop()
                known_wrapper = (
                    stack == [WRAPPER, "data"]
                    or (len(stack) == 5 and stack[0] is WRAPPER
                        and stack[1] is MARK and stack[2] == "prep_id"
                        and isinstance(stack[3], str) and stack[4] == "data")
                )
                if root_seen or cls not in CLASSES or args != () or (stack and not known_wrapper):
                    raise IdentityError("Unsupported object header")
                root_seen = True
                root_index = len(stack)
                stack.append(ROOT)
            elif op == "EMPTY_DICT":
                if not stack and not root_seen:
                    stack.append(WRAPPER)
                elif root_seen and len(stack) == root_index + 1 and stack[-1] is ROOT:
                    stack.append(STATE)
                else:
                    raise IdentityError("Unsupported state header")
            elif op == "MARK":
                stack.append(MARK)
            elif op == "EMPTY_LIST":
                stack.append([])
            elif op == "APPEND":
                value = stack.pop()
                if not isinstance(stack[-1], list) or not isinstance(value, str):
                    raise IdentityError("Encoder must contain strings")
                stack[-1].append(value)
            elif op == "APPENDS":
                index = next(i for i in range(len(stack) - 1, -1, -1) if stack[i] is MARK)
                values = stack[index + 1:]
                if not isinstance(stack[index - 1], list) or not all(isinstance(x, str) for x in values):
                    raise IdentityError("Encoder must contain strings")
                stack[index - 1].extend(values)
                del stack[index:]
            elif op in {"BINPUT", "LONG_BINPUT", "PUT", "MEMOIZE"}:
                memo[len(memo) if op == "MEMOIZE" else int(arg)] = stack[-1]
            elif op in {"BINGET", "LONG_BINGET", "GET"}:
                stack.append(memo[int(arg)])
            else:
                raise IdentityError("Unsupported opcode before identity boundary: " + op)
            # The exact project state starts with scalar settings and encoder lists.
            # A new top-level key proves all previous list batches are complete.
            if (root_seen and len(stack) >= root_index + 4
                    and stack[root_index] is ROOT and stack[root_index + 1] is STATE
                    and stack[root_index + 2] is MARK and (len(stack) - root_index) % 2 == 0
                    and isinstance(stack[-1], str) and stack[-1] == "uid"):
                pairs = stack[root_index + 3:-1]
                keys = pairs[::2]
                if keys != ["hist_len", "seed", "users", "items", "categories", "brands"]:
                    raise IdentityError("Unexpected state fields before event arrays")
                data = dict(zip(keys, pairs[1::2]))
                for field in FIELDS:
                    values = data[field]
                    if (not isinstance(values, list) or len(values) < 2
                            or values[:2] != ["__PAD__", "__UNKNOWN__"]
                            or not all(isinstance(x, str) for x in values)
                            or any(a >= b for a, b in zip(values[2:], values[3:]))
                            or any(x in {"__PAD__", "__UNKNOWN__"} for x in values[2:])):
                        raise IdentityError("Invalid encoder list: " + field)
                return {field: data[field] for field in FIELDS}, stream.tell()
        raise IdentityError("Missing complete encoder prefix")
    except (KeyError, IndexError, StopIteration, TypeError, UnicodeError) as exc:
        raise IdentityError("Malformed identity prefix") from exc


def extract(source, expected_sha256=None, expected_fingerprint=None):
    source = Path(source)
    with source.open("rb") as stream:
        before = os.fstat(stream.fileno())
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        sha256 = digest.hexdigest()
        if expected_sha256 and sha256 != expected_sha256:
            raise IdentityError("Source SHA256 mismatch")
        stream.seek(0)
        encoders, prefix_bytes = parse_prefix(stream)
        after = os.fstat(stream.fileno())
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise IdentityError("Source changed during extraction")
    encoder_hash = fingerprint([encoders[field] for field in FIELDS])
    if expected_fingerprint and encoder_hash != expected_fingerprint:
        raise IdentityError("Encoder fingerprint mismatch")
    return {"format": "project-pickle-identity-v1", "source": str(source.resolve()),
            "source_sha256": sha256, "source_bytes": before.st_size,
            "parsed_prefix_bytes": prefix_bytes, "encoders_hash": encoder_hash,
            "counts": {field: len(encoders[field]) for field in FIELDS},
            "reserved_indices": {"pad": 0, "unknown": 1}, "users": encoders["users"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--expected-fingerprint")
    args = parser.parse_args()
    try:
        if args.source.resolve() == args.output.resolve():
            raise IdentityError("Output must differ from source")
        result = extract(args.source, args.expected_sha256, args.expected_fingerprint)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=args.output.parent,
                                         prefix=".identity-", delete=False) as stream:
            temporary = Path(stream.name)
            try:
                json.dump(result, stream, ensure_ascii=True)
                stream.write("\n")
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        try:
            os.replace(temporary, args.output)
        finally:
            temporary.unlink(missing_ok=True)
        print(json.dumps({key: value for key, value in result.items() if key != "users"}))
    except (OSError, ValueError) as exc:
        detail = str(exc) if isinstance(exc, IdentityError) else type(exc).__name__
        parser.exit(1, "Identity extraction failed: " + detail + "\n")


if __name__ == "__main__":
    main()
