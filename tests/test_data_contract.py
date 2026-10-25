import unittest,tempfile,json
from pathlib import Path
import numpy as np
from sav_bandnet.data import crop,Windows
from sav_bandnet.infer import si
class Contracts(unittest.TestCase):
 def test_startup_and_tail(self):
  x=np.arange(10000,dtype=np.float32)[None]
  first=crop(x,-6400,16000);np.testing.assert_array_equal(first[:,6400:12800],x[:,:6400]);self.assertTrue((first[:,:6400]==0).all())
  tail=crop(x,0,16000);np.testing.assert_array_equal(tail[:,6400:10000],x[:,6400:]);self.assertTrue((tail[:,10000:]==0).all())
 def test_visual_axis(self):
  frames=np.arange(25*3).reshape(25,3);y=crop(frames,-10,25,0);self.assertEqual(y.shape,(25,3));np.testing.assert_array_equal(y[10:],frames[:15])
 def test_all_scene_coverage(self):
  with tempfile.TemporaryDirectory() as root:
   m=Path(root)/'manifests';m.mkdir();(m/'tr.jsonl').write_text(''.join(json.dumps({'seconds':v})+'\n' for v in [1.,2.,3.]))
   d=Windows(root,'tr');self.assertEqual({i for i,s in d.plan},{0,1,2});self.assertTrue(all(any(i==j and s==-6400 for i,s in d.plan) for j in range(3)))
 def test_si_scale_invariant(self):
  r=np.sin(np.arange(4000)*.03);noise=np.cos(np.arange(4000)*.11)*.2;y=r+noise
  self.assertAlmostEqual(si(y,r),si(2*y,r),places=8);self.assertIsNone(si(y,np.zeros_like(r)))
if __name__=='__main__':unittest.main()
