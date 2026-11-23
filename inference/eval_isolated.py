
import os,runpy,sys
from pathlib import Path
source=Path(__file__).resolve().parent
sys.path.insert(0,str(source))
from seeded_world import SeededWorld
original=SeededWorld.evaluate
def isolated(self,*args,**kwargs):
    directory=Path(os.environ['PAPER_EVAL_VIDEO_DIR'])
    directory.mkdir(parents=True,exist_ok=True)
    kwargs['video']=directory
    return original(self,*args,**kwargs)
SeededWorld.evaluate=isolated
runpy.run_path(str(source/'eval.py'),run_name='__main__')
