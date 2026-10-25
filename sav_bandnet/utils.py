from pathlib import Path
import hashlib,json,os
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
 return h.hexdigest()

def state_sha(state):
 h=hashlib.sha256()
 for n,v in sorted(state.items()):
  h.update(n.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
