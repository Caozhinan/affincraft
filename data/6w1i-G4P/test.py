import pymol2

with pymol2.PyMOL() as pymol:
    pymol.cmd.load("ligand.sdf")
    pymol.cmd.zoom()
    pymol.cmd.show("sticks")
    pymol.cmd.png("ligand.png", width=1200, height=900, dpi=300, ray=1)