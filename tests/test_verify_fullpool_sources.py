"""Lightweight full-pool provenance tests; no builder, model or targets."""
import copy
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import verify_fullpool_sources as gate


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


class FullPoolSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve()

    def test_unsafe_receipt_and_missing_readiness_leave_no_output(self):
        readonly=self.root/'input';readonly.mkdir()
        (self.root/'alias').symlink_to(readonly,target_is_directory=True)
        (self.root/'dangling').symlink_to(self.root/'missing')
        for path in (readonly/'new.json',self.root/'alias/new.json',self.root/'dangling',self.root/'a/../new.json'):
            with self.subTest(path=path):
                with self.assertRaises(ValueError):gate.output_path(path,[readonly])
        output=self.root/'gates/new.json'
        with self.assertRaisesRegex(ValueError,'prerequisite not ready'):gate.verify(self.root,output)
        self.assertFalse(output.parent.exists())

    def fixture_transfer(self):
        local=self.root/'full';write(local/'COMPLETED.json',{'status':'complete'});write(local/'payload.json',{'value':1})
        files={name:{'bytes':path.stat().st_size,'sha256':gate.sha(path)} for name,path in gate.tree_files(local).items()}
        inventory={'root':gate.FULL,'marker':'COMPLETED.json','files':files}
        receipt_path=self.root/'transfer_verified.json';ip=receipt_path.with_suffix('.inventory.json');write(ip,inventory)
        receipt=dict(status='verified',local_root=str(local),remote_root=gate.FULL,inventory_sha256=gate.sha(ip),completion_sha256=gate.sha(local/'COMPLETED.json'),files=len(files),bytes=sum(x['bytes'] for x in files.values()),remote_inventory_before_after_equal=True,local_exact_coverage=True,local_sha_size_equal=True,remote_mutated=False)
        write(receipt_path,receipt)
        return local,receipt_path,receipt,ip,inventory

    def test_transfer_binds_roles_markers_seals_flags_and_payload_bytes(self):
        local,rp,receipt,ip,inventory=self.fixture_transfer()
        def verify():return gate.transfer(rp,local,gate.FULL,'COMPLETED.json')
        self.assertEqual(verify(),inventory['files'])
        mutations={'local_root':str(self.root),'remote_root':gate.REMOTE,'inventory_sha256':'0'*64,'completion_sha256':'0'*64,'files':5,'bytes':0,'remote_inventory_before_after_equal':False,'local_exact_coverage':False,'local_sha_size_equal':False,'remote_mutated':True}
        for key,value in mutations.items():
            altered=dict(receipt);altered[key]=value;write(rp,altered)
            with self.subTest(receipt=key):
                with self.assertRaises(ValueError):verify()
        write(rp,receipt)
        for key,value in [('root',gate.REMOTE),('marker','control_v1/COMPLETED.json')]:
            altered=copy.deepcopy(inventory);altered[key]=value;write(ip,altered)
            changed=dict(receipt);changed['inventory_sha256']=gate.sha(ip);write(rp,changed)
            with self.subTest(inventory=key):
                with self.assertRaises(ValueError):verify()
        write(ip,inventory);write(rp,receipt)
        (local/'payload.json').write_text((local/'payload.json').read_text().replace('1','2'))
        with self.assertRaisesRegex(ValueError,'SHA/size'):verify()

    def test_launch_rejects_worker_source_mode_and_policy_changes(self):
        launch={'command':list(gate.COMMAND),'interval_seconds':900,'automatic_restart':False,'test_allowed':False}
        gate.check_launch(launch)
        for index in range(len(gate.COMMAND)):
            changed=copy.deepcopy(launch);changed['command'][index]='changed'
            with self.subTest(index=index):
                with self.assertRaises(ValueError):gate.check_launch(changed)
        for key,value in [('interval_seconds',901),('automatic_restart',True),('test_allowed',True)]:
            changed=copy.deepcopy(launch);changed[key]=value
            with self.assertRaises(ValueError):gate.check_launch(changed)

    def test_control_status_requires_complete_exact_final_fields(self):
        status=dict(utc='2026-09-25',pid=1,returncode=0,elapsed_seconds=10,free_bytes=100,
                    memory_current=10,memory_max=100,completed=True,log_tail='done')
        control=dict(status,status='complete',peak_child_rss_kib=20)
        gate.check_control(control,status)
        with self.assertRaises(ValueError):gate.check_control(control,{})
        for key in status:
            changed=dict(status);changed.pop(key)
            with self.subTest(missing=key):
                with self.assertRaises(ValueError):gate.check_control(control,changed)
        changed=dict(status);changed['pid']=2
        with self.assertRaises(ValueError):gate.check_control(control,changed)
        changed=dict(control);changed['status']='failed'
        with self.assertRaises(ValueError):gate.check_control(changed,status)

    def test_resealed_supervisor_rejected_even_with_matching_local_remote(self):
        archive=self.root/'archive';remote=archive/'remote_root'
        for path in (archive/'supervise_full.py',remote/'supervise_full.py'):
            path.parent.mkdir(parents=True,exist_ok=True);path.write_text('modified supervisor')
        self.assertEqual(gate.sha(archive/'supervise_full.py'),gate.sha(remote/'supervise_full.py'))
        with self.assertRaisesRegex(ValueError,'frozen supervisor'):gate.check_supervisor(archive,remote)

    def test_real_frozen_pilot_matches_tar_extraction_and_prior_completion(self):
        archive=Path(__file__).resolve().parents[1]/'server_snapshot/full_prefix_pools_20260925'
        pilot=archive/'pilot_extracted'/gate.PILOT
        if not (pilot/'COMPLETED.json').exists():self.skipTest('optional local sealed pilot unavailable')
        semantic={'pilot_completion_sha256':gate.PILOT_SHA,'pilot_files':{name:{'bytes':p.stat().st_size,'sha256':gate.sha(p)} for name,p in gate.tree_files(pilot).items()}}
        gate.check_pilot(archive,pilot,semantic)
        gate.check_completion(gate.read(pilot/'COMPLETED.json'),'pilot')
        for name in semantic['pilot_files']:
            altered=copy.deepcopy(semantic);altered['pilot_files'][name]['sha256']='0'*64
            with self.subTest(name=name):
                with self.assertRaises(ValueError):gate.check_pilot(archive,pilot,altered)

    def test_pilot_tamper_and_unsafe_tar_are_rejected(self):
        archive=self.root/'archive';pilot=archive/'pilot_extracted'/gate.PILOT
        write(pilot/'COMPLETED.json',{'status':'complete'})
        write(archive/'pilot_COMPLETED.json',{'status':'complete'})
        payload=pilot/'COMPLETED.json';digest=gate.sha(payload)
        semantic={'pilot_completion_sha256':digest,'pilot_files':{'COMPLETED.json':{'bytes':payload.stat().st_size,'sha256':digest}}}
        def pack(name,symlink=False):
            with tarfile.open(archive/'pilot_archive.tar.gz','w:gz') as out:
                root=tarfile.TarInfo(gate.PILOT);root.type=tarfile.DIRTYPE;out.addfile(root)
                item=tarfile.TarInfo(name)
                if symlink:item.type=tarfile.SYMTYPE;item.linkname='/outside';out.addfile(item)
                else:item.size=payload.stat().st_size;out.addfile(item,io.BytesIO(payload.read_bytes()))
            write(archive/'pilot_extraction_receipt.json',dict(status='verified',local_completion_matches_previously_captured=True,completion_sha256=digest,tar_sha256=gate.sha(archive/'pilot_archive.tar.gz'),safe_members=2))
        with patch.object(gate,'PILOT_SHA',digest):
            pack(gate.PILOT+'/COMPLETED.json');gate.check_pilot(archive,pilot,semantic)
            for name,link in [(gate.PILOT+'/../escape',False),('/absolute',False),(gate.PILOT+'/COMPLETED.json',True)]:
                pack(name,link)
                with self.subTest(name=name,symlink=link):
                    with self.assertRaises(ValueError):gate.check_pilot(archive,pilot,semantic)


if __name__=='__main__':unittest.main()
