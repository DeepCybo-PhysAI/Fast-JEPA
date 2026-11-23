
import argparse,base64,csv,hashlib,importlib,importlib.metadata,json,sys
p=argparse.ArgumentParser();p.add_argument('--gpu',action='store_true');a=p.parse_args()
modules=['torch','torchvision','hydra','omegaconf','lightning','stable_pretraining','stable_worldmodel','h5py','hdf5plugin','lance','zstandard','pygame','cv2']
loaded={name:importlib.import_module(name) for name in modules}
pygame=loaded['pygame'];assert pygame.version.ver=='2.6.1' and not getattr(pygame,'IS_CE',False),'Wrong pygame renderer: rerun scripts/install.sh'

dist=importlib.metadata.distribution('opencv-python')
for file,digest,size in csv.reader(dist.read_text('RECORD').splitlines()):
    if file.startswith('cv2/') and file.endswith('.so') and digest:
        h=hashlib.sha256()
        with dist.locate_file(file).open('rb') as f:
            for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
        assert base64.urlsafe_b64encode(h.digest()).rstrip(b'=').decode()==digest.split('=',1)[1],'Wrong cv2 backend: rerun scripts/install.sh'
report={'python':sys.version.split()[0],'torch':loaded['torch'].__version__,'imports':'PASS','gpu_checked':a.gpu,'pygame':pygame.version.ver,'pygame_ce_active':False,'opencv_backend':'opencv-python','mujoco_distribution':importlib.metadata.version('mujoco')}
if a.gpu:
    t=loaded['torch'];assert t.cuda.is_available(),'A working CUDA GPU is required'
    x=t.ones(4,device='cuda');assert x.sum().item()==4
    try:
        mujoco=importlib.import_module('mujoco');c=mujoco.GLContext(32,32);c.make_current();c.free()
    except Exception as ex:raise SystemExit('MuJoCo/EGL unavailable: install the system libraries and NVIDIA graphics runtime described in README.md (Environment setup)') from ex
    report.update(device=t.cuda.get_device_name(),cuda=t.version.cuda,egl='PASS')
print(json.dumps(report,indent=2))
