"""Bounded-memory, fail-closed candidate storage for cross protocol v2."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
from collections.abc import Sequence
import numpy as np

CACHE_VERSION = 'cross-pools-int32-v2'
BLOCK_ROWS = 1024


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def rows_hash(rows):
    """Same canonical JSON array hash as the legacy implementation, streamed."""
    digest = hashlib.sha256(); digest.update(b'[')
    for index, row in enumerate(rows):
        if index: digest.update(b',')
        digest.update(json.dumps(list(row), ensure_ascii=True, separators=(',', ':')).encode())
    digest.update(b']')
    return digest.hexdigest()


class PositionRecords(Sequence):
    """A record view over immutable positions, without millions of Python tuples."""
    def __init__(self, data, positions):
        self.data, self.positions = data, positions
    def __len__(self): return len(self.positions)
    def __getitem__(self, index):
        if isinstance(index, slice):
            return PositionRecords(self.data, self.positions[index])
        position = int(self.positions[index]); uid = int(self.data.uid[position])
        return uid, position, str(self.data.users[uid])


class PoolRows(Sequence):
    def __init__(self, items, lengths, metadata):
        self.items, self.lengths, self.metadata = items, lengths, metadata
    def __len__(self): return len(self.lengths)
    def __getitem__(self, index):
        if isinstance(index, slice): return [self[i] for i in range(*index.indices(len(self)))]
        return self.items[index, :int(self.lengths[index])]


def pool_hash(pools):
    return rows_hash((int(item) for item in row) for row in pools)


def _safe_file(directory, name):
    path = (directory / name).resolve()
    if path.parent != directory.resolve(): raise ValueError('cache path escapes directory')
    return path


def load_cache(path, source=None):
    path = Path(path)
    obj = json.loads(path.read_text())
    if obj.get('schema') != CACHE_VERSION or obj.get('status') != 'complete':
        raise ValueError('candidate cache is incomplete or unsupported')
    if source is not None and obj.get('source') != source:
        raise ValueError('candidate cache provenance mismatch')
    if obj.get('dtype') != '<i4' or obj.get('lengths_dtype') != '<i4':
        raise ValueError('candidate cache dtype mismatch')
    bound = obj.get('source', {})
    if (obj.get('records_hash') != bound.get('records_hash') or obj.get('position_uid_row_hash') != bound.get('records_hash')
            or obj.get('budget') != obj.get('shape', [None,None])[1]
            or ('budget' in bound and obj.get('budget') != bound['budget'])):
        raise ValueError('candidate cache metadata mismatch')
    if 'records_count' in bound and obj['shape'][0] != bound['records_count']:
        raise ValueError('candidate cache row count mismatch')
    arrays = {}
    for name in ('items', 'lengths'):
        receipt = obj['files'][name]
        asset = _safe_file(path.parent, receipt['name'])
        if file_sha256(asset) != receipt['sha256']: raise ValueError('candidate cache content hash mismatch')
        arrays[name] = np.load(asset, mmap_mode='r', allow_pickle=False)
        if arrays[name].dtype.str != '<i4': raise ValueError('candidate cache array dtype mismatch')
    items, lengths = arrays['items'], arrays['lengths']
    if list(items.shape) != obj['shape'] or lengths.shape != (len(items),):
        raise ValueError('candidate cache shape mismatch')
    for start in range(0, len(items), BLOCK_ROWS):
        lens = lengths[start:start+BLOCK_ROWS]
        if np.any(lens < 0) or np.any(lens > items.shape[1]): raise ValueError('candidate cache invalid lengths')
        for offset, length in enumerate(lens):
            row = items[start+offset]; valid = row[:int(length)]
            if np.any(valid < 2) or ('catalog_size' in bound and np.any(valid >= bound['catalog_size'])) or len(np.unique(valid)) != len(valid) or np.any(row[int(length):] != 0):
                raise ValueError('candidate cache invalid IDs, duplicates or padding')
    pools = PoolRows(items, lengths, obj)
    if pool_hash(pools) != obj['pool_hash']: raise ValueError('candidate cache logical hash mismatch')
    return pools, obj


def build_cache(path, records, budget, source, build_block, block_rows=BLOCK_ROWS):
    """Publish sidecar last; interrupted payloads are never reusable or erased."""
    path = Path(path)
    if path.exists(): return load_cache(path, source)
    if len(records) < 1 or budget < 1 or block_rows < 1: raise ValueError('invalid candidate cache shape')
    names = {name: path.with_name(path.stem + '.' + name + '.npy') for name in ('items', 'lengths')}
    temporary = {name: p.with_suffix(p.suffix + '.incomplete') for name, p in names.items()}
    if any(p.exists() for p in [*names.values(), *temporary.values()]):
        raise FileExistsError('incomplete candidate cache exists; preserve and inspect it before retry')
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(path.suffix + '.building')
    with lock.open('x') as stream:
        stream.write('exclusive candidate builder; retained as construction evidence\n')
        stream.flush(); os.fsync(stream.fileno())
    items = np.lib.format.open_memmap(temporary['items'], mode='w+', dtype='<i4', shape=(len(records), budget))
    lengths = np.lib.format.open_memmap(temporary['lengths'], mode='w+', dtype='<i4', shape=(len(records),))
    for start in range(0, len(records), block_rows):
        stop = min(start + block_rows, len(records))
        rows = build_block(records[start:stop])
        if len(rows) != stop-start: raise ValueError('candidate builder row mismatch')
        items[start:stop] = 0
        for offset, row in enumerate(rows):
            ids = [int(i) for i in row]
            if len(ids)>budget or len(set(ids)) != len(ids) or any(i<2 or i>np.iinfo(np.int32).max for i in ids):
                raise ValueError('invalid candidate row')
            items[start+offset, :len(ids)] = ids
            lengths[start+offset] = len(ids)
        items.flush(); lengths.flush()
    logical = pool_hash(PoolRows(items, lengths, {}))
    del items, lengths
    for name in names:
        with temporary[name].open('rb') as stream: os.fsync(stream.fileno())
        os.replace(temporary[name], names[name])
    obj = dict(schema=CACHE_VERSION, status='complete', dtype='<i4', lengths_dtype='<i4',
               shape=[len(records), budget], budget=budget, block_rows=block_rows,
               records_hash=source['records_hash'], position_uid_row_hash=source['records_hash'],
               pool_hash=logical, source=source,
               files={name: dict(name=p.name, sha256=file_sha256(p)) for name,p in names.items()})
    temp_sidecar = path.with_suffix(path.suffix + '.incomplete')
    with temp_sidecar.open('x') as stream:
        json.dump(obj, stream, sort_keys=True, indent=2); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp_sidecar, path)
    return load_cache(path, source)
