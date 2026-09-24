"""Focused adversarial tests for local source verification (no training/scoring)."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import verify_early_cross_sources as cross
import verify_final_source_mapping as final


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class SourceGuards(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def test_receipt_rejects_traversal_symlinks_existing_and_readonly(self):
        readonly = self.root / 'archive'
        readonly.mkdir()
        (self.root / 'alias').symlink_to(readonly, target_is_directory=True)
        (self.root / 'dangling').symlink_to(self.root / 'absent')
        existing = self.root / 'existing.json'
        existing.write_text('immutable')
        for module in (cross, final):
            for path in (readonly/'new.json', readonly/'nested/new.json', existing,
                         self.root/'alias/new.json', self.root/'dangling',
                         self.root/'unused/../new.json'):
                with self.subTest(module=module.__name__, path=path):
                    with self.assertRaises((ValueError, RuntimeError)):
                        module.checked_receipt(path, [readonly])
            self.assertEqual(module.checked_receipt(self.root/'gates/new.json', [readonly]),
                             self.root/'gates/new.json')
        self.assertEqual(existing.read_text(), 'immutable')
        self.assertFalse((readonly/'new.json').exists())

    def test_mapped_input_is_immutable_and_source_symlinks_fail(self):
        path = self.root/'archive/input.json'
        write_json(path, {'source': True})
        digest = cross.sha(path)
        used = {}
        self.assertEqual(final.checked_local(path, digest, used), path)
        alias = self.root/'alias.json'
        alias.symlink_to(path)
        with self.assertRaises((ValueError, RuntimeError)):
            final.checked_local(alias, digest, {})
        with self.assertRaises((ValueError, RuntimeError)):
            final.checked_receipt(path, [path.parent])
        with self.assertRaises((ValueError, RuntimeError)):
            final.checked_receipt(path.parent/'new.json', [path.parent])

    def test_fixed_source_set_rejects_missing_extra_and_resealed_runner(self):
        sources = dict(cross.FROZEN_SOURCES)
        with patch.object(cross, 'sha', side_effect=lambda path: sources[path.name]):
            cross.check_sources(sources, self.root, self.root)
        bads = []
        missing = dict(sources); missing.pop('monitor_training_diagnostics.py'); bads.append(missing)
        extra = dict(sources); extra['extra.py'] = '0'*64; bads.append(extra)
        for name in sources:
            resealed = dict(sources); resealed[name] = '0'*64; bads.append(resealed)
        for bad in bads:
            with self.subTest(keys=list(bad)):
                # Even matching deployed bytes/run-manifest cannot change pinned identities.
                with patch.object(cross, 'sha', side_effect=lambda path: bad[path.name]):
                    with self.assertRaises((ValueError, RuntimeError)):
                        cross.check_sources(bad, self.root, self.root)

    def transfer_fixture(self, control):
        local = self.root/('remote_root' if control else 'run')
        marker = 'control_v1/COMPLETED.json' if control else 'COMPLETED.json'
        remote = '/root/early_stop_cross_20260925' if control else '/root/autodl-tmp/early_stop_cross_20260925_v1'
        write_json(local/marker, {'status': 'complete', 'returncode': 0, 'failed_marker': False})
        write_json(local/'src/source.json', {'frozen': 1})
        inventory = {'root': remote, 'marker': marker, 'files': {
            str(p.relative_to(local)): {'sha256': cross.sha(p), 'bytes': p.stat().st_size}
            for p in local.rglob('*') if p.is_file()}}
        receipt_path = self.root/('control_transfer_verified.json' if control else 'transfer_verified.json')
        inventory_path = receipt_path.with_suffix('.inventory.json')
        write_json(inventory_path, inventory)
        digest = cross.sha(local/marker)
        receipt = {'status':'verified', 'local_root':str(local), 'remote_root':remote,
                   'inventory_sha256':cross.sha(inventory_path), 'completion_sha256':digest,
                   'files':len(inventory['files']), 'bytes':sum(x['bytes'] for x in inventory['files'].values()),
                   'remote_inventory_before_after_equal':True, 'local_exact_coverage':True,
                   'local_sha_size_equal':True, 'remote_mutated':False}
        if not control: receipt['cross_completion_sha256'] = digest
        write_json(receipt_path, receipt)
        return local, marker, remote, receipt_path, inventory_path, receipt, inventory

    def test_transfer_receipt_and_inventory_bindings(self):
        for control in (False, True):
            local, marker, remote, receipt_path, inventory_path, receipt, inventory = self.transfer_fixture(control)
            def verify():
                return cross.check_transfer(receipt_path, local, remote, marker,
                    None if control else cross.sha(local/marker), rehash=control)
            verify()
            mutations = [('local_root', str(self.root/'other')), ('remote_root', '/other'),
                         ('inventory_sha256', '0'*64), ('completion_sha256','0'*64),
                         ('files', 500), ('bytes', 0), ('remote_inventory_before_after_equal',False),
                         ('local_exact_coverage',False), ('local_sha_size_equal',False), ('remote_mutated',True)]
            if not control: mutations.append(('cross_completion_sha256','0'*64))
            for key, value in mutations:
                changed = dict(receipt); changed[key] = value; write_json(receipt_path, changed)
                with self.subTest(control=control, receipt_key=key):
                    with self.assertRaises((ValueError, RuntimeError)): verify()
            write_json(receipt_path, receipt)
            for key, value in [('root','/wrong'), ('marker','wrong.json')]:
                changed = copy.deepcopy(inventory); changed[key] = value
                write_json(inventory_path, changed)
                resealed = dict(receipt); resealed['inventory_sha256'] = cross.sha(inventory_path)
                write_json(receipt_path, resealed)
                with self.subTest(control=control, inventory_key=key):
                    with self.assertRaises((ValueError, RuntimeError)): verify()
            write_json(inventory_path, inventory); write_json(receipt_path, receipt)
            payload = local/'src/source.json'
            payload.write_text(payload.read_text().replace('1','2'))
            if control:
                with self.assertRaises((ValueError, RuntimeError)): verify()
            write_json(payload, {'frozen':1})
            (local/marker).write_text((local/marker).read_text().replace('complete','rejected'))
            with self.assertRaises((ValueError, RuntimeError)): verify()

    def test_control_launch_is_bound_to_frozen_run_and_sources(self):
        source = '/root/early_stop_cross_20260925/src/'
        launch = {
            'run_dir':'/root/autodl-tmp/early_stop_cross_20260925_v1',
            'control_dir':'/root/early_stop_cross_20260925/control_v1',
            'automatic_restart':False, 'shutdown_allowed':False,
            'command':['/root/miniconda3/bin/python', source+'run_early_stop_cross.py',
                '--legacy-source-dir','/root/training_diagnostics_20260924/src_v2/code',
                '--protocol-manifest','/root/training_diagnostics_20260924/protocol_v2/protocol_manifest.json',
                '--raw-run','/root/autodl-tmp/training_diagnostics_20260924_raw_v1',
                '--old-zero-run','/root/autodl-tmp/next_cross_20260924_dev',
                '--run-dir','/root/autodl-tmp/early_stop_cross_20260925_v1',
                '--protocol-doc',source+'OVERNIGHT_CROSS_PROTOCOL_20260925.md',
                '--archive-receipt',source+'archive_verified.json',
                '--backup-receipt',source+'RECOVERY_BACKUP_BINDING.json', '--device','cuda']}
        cross.check_control_launch(launch)
        for key in ('run_dir','control_dir','automatic_restart','shutdown_allowed'):
            changed = copy.deepcopy(launch)
            changed[key] = True if isinstance(changed[key],bool) else '/wrong'
            with self.subTest(key=key):
                with self.assertRaises((ValueError, RuntimeError)): cross.check_control_launch(changed)
        for index in range(1, len(launch['command'])):
            changed = copy.deepcopy(launch); changed['command'][index] = '/wrong'
            with self.subTest(command_index=index):
                with self.assertRaises((ValueError, RuntimeError)): cross.check_control_launch(changed)

    def array_fixture(self):
        run = self.root/'arrays'
        completed = {'seeds':dict.fromkeys(['42','43','44']), 'evidence_hashes':{}}
        for seed in ('42','43','44'):
            for filename in ('screen_step1','screen_step2','screen_step3','confirm_best'):
                path = run/f'seed_{seed}/zero_cross/{filename}_users.npz'
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(path, uid=[1], position=[2], candidate_ids=[[1,2]], lengths=[2], predictions=[0.2])
                completed['evidence_hashes'][str(path.relative_to(run))] = cross.sha(path)
        return run, completed

    def test_all_arrays_sealed_before_read_even_when_identities_unchanged(self):
        run, completed = self.array_fixture()
        paths = cross.checked_arrays(run, completed, 'screen', 3) + cross.checked_arrays(run, completed, 'confirm', 1)
        self.assertEqual(len(paths), 12)
        for path in paths:
            original = path.read_bytes()
            np.savez(path, uid=[1], position=[2], candidate_ids=[[1,2]], lengths=[2], predictions=[0.9])
            split = 'screen' if path.name.startswith('screen') else 'confirm'
            with self.subTest(path=path):
                with self.assertRaises((ValueError, RuntimeError)):
                    cross.checked_arrays(run, completed, split, 3 if split=='screen' else 1)
            path.write_bytes(original)

    def test_aggregate_array_count_cannot_hide_wrong_seed_distribution(self):
        run, completed = self.array_fixture()
        old = run/'seed_42/zero_cross/screen_step1_users.npz'
        new = run/'seed_43/zero_cross/screen_step4_users.npz'
        old.rename(new)
        completed['evidence_hashes'][str(new.relative_to(run))] = completed['evidence_hashes'].pop(str(old.relative_to(run)))
        with self.assertRaises((ValueError, RuntimeError)):
            cross.checked_arrays(run, completed, 'screen', 3)


if __name__ == '__main__':
    unittest.main()
