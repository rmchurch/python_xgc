
from mesh_xgc import mesh_xgc

pfilename = 'LiInjection_157267/p157267.02290' 
gfilename = 'LiInjection_157267/g157267.02290' 
mobj = mesh_xgc(pfilename,gfilename)

mobj.plot_spacing()
mobj.write_spacing()
