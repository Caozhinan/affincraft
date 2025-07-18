##############################################################################
# MC-shell I/O capture file.
# Creation Date and Time:  Mon Jul  7 16:08:26 2025

##############################################################################
Hello world from PE 0
Vnm_tstart: starting timer 26 (APBS WALL CLOCK)..
NOsh_parseInput:  Starting file parsing...
NOsh: Parsing READ section
NOsh: Storing molecule 0 path complex_A
NOsh: Done parsing READ section
NOsh: Done parsing READ section (nmol=1, ndiel=0, nkappa=0, ncharge=0, npot=0)
NOsh: Parsing ELEC section
NOsh_parseMG: Parsing parameters for MG calculation
NOsh_parseMG:  Parsing dime...
PBEparm_parseToken:  trying dime...
MGparm_parseToken:  trying dime...
NOsh_parseMG:  Parsing cglen...
PBEparm_parseToken:  trying cglen...
MGparm_parseToken:  trying cglen...
NOsh_parseMG:  Parsing fglen...
PBEparm_parseToken:  trying fglen...
MGparm_parseToken:  trying fglen...
NOsh_parseMG:  Parsing cgcent...
PBEparm_parseToken:  trying cgcent...
MGparm_parseToken:  trying cgcent...
NOsh_parseMG:  Parsing fgcent...
PBEparm_parseToken:  trying fgcent...
MGparm_parseToken:  trying fgcent...
NOsh_parseMG:  Parsing mol...
PBEparm_parseToken:  trying mol...
NOsh_parseMG:  Parsing lpbe...
PBEparm_parseToken:  trying lpbe...
NOsh: parsed lpbe
NOsh_parseMG:  Parsing bcfl...
PBEparm_parseToken:  trying bcfl...
NOsh_parseMG:  Parsing pdie...
PBEparm_parseToken:  trying pdie...
NOsh_parseMG:  Parsing sdie...
PBEparm_parseToken:  trying sdie...
NOsh_parseMG:  Parsing srfm...
PBEparm_parseToken:  trying srfm...
NOsh_parseMG:  Parsing chgm...
PBEparm_parseToken:  trying chgm...
MGparm_parseToken:  trying chgm...
NOsh_parseMG:  Parsing sdens...
PBEparm_parseToken:  trying sdens...
NOsh_parseMG:  Parsing srad...
PBEparm_parseToken:  trying srad...
NOsh_parseMG:  Parsing swin...
PBEparm_parseToken:  trying swin...
NOsh_parseMG:  Parsing temp...
PBEparm_parseToken:  trying temp...
NOsh_parseMG:  Parsing calcenergy...
PBEparm_parseToken:  trying calcenergy...
NOsh_parseMG:  Parsing calcforce...
PBEparm_parseToken:  trying calcforce...
NOsh_parseMG:  Parsing write...
PBEparm_parseToken:  trying write...
NOsh_parseMG:  Parsing end...
MGparm_check:  checking MGparm object of type 1.
NOsh:  nlev = 6, dime = (129, 129, 129)
NOsh: Done parsing ELEC section (nelec = 1)
NOsh: Parsing PRINT section
NOsh: Done parsing PRINT section
NOsh: Done parsing PRINT section
NOsh: Done parsing file (got QUIT)
Valist_readPQR: Counted 2983 atoms
Valist_getStatistics:  Max atom coordinate:  (17.603, 62.851, 59.089)
Valist_getStatistics:  Min atom coordinate:  (-30.195, 19.712, 12.672)
Valist_getStatistics:  Molecule center:  (-6.296, 41.2815, 35.8805)
NOsh_setupCalcMGAUTO(/home/ubuntu/git/apbs-pdb2pqr/apbs/src/generic/nosh.c, 1855):  coarse grid center = -6.296 41.2815 35.8805
NOsh_setupCalcMGAUTO(/home/ubuntu/git/apbs-pdb2pqr/apbs/src/generic/nosh.c, 1860):  fine grid center = -6.296 41.2815 35.8805
NOsh_setupCalcMGAUTO (/home/ubuntu/git/apbs-pdb2pqr/apbs/src/generic/nosh.c, 1872):  Coarse grid spacing = 0.669521, 0.607763, 0.650555
NOsh_setupCalcMGAUTO (/home/ubuntu/git/apbs-pdb2pqr/apbs/src/generic/nosh.c, 1874):  Fine grid spacing = 0.550086, 0.513758, 0.53893
NOsh_setupCalcMGAUTO (/home/ubuntu/git/apbs-pdb2pqr/apbs/src/generic/nosh.c, 1876):  Displacement between fine and coarse grids = 0, 0, 0
NOsh:  2 levels of focusing with 0.821611, 0.845326, 0.828415 reductions
NOsh_setupMGAUTO:  Resetting boundary flags
NOsh_setupCalcMGAUTO (/home/ubuntu/git/apbs-pdb2pqr/apbs/src/generic/nosh.c, 1970):  starting mesh repositioning.
NOsh_setupCalcMGAUTO (/home/ubuntu/git/apbs-pdb2pqr/apbs/src/generic/nosh.c, 1972):  coarse mesh center = -6.296 41.2815 35.8805
NOsh_setupCalcMGAUTO (/home/ubuntu/git/apbs-pdb2pqr/apbs/src/generic/nosh.c, 1977):  coarse mesh upper corner = 36.5534 80.1783 77.5161
NOsh_setupCalcMGAUTO (/home/ubuntu/git/apbs-pdb2pqr/apbs/src/generic/nosh.c, 1982):  coarse mesh lower corner = -49.1454 2.38465 -5.75505
NOsh_setupCalcMGAUTO (/home/ubuntu/git/apbs-pdb2pqr/apbs/src/generic/nosh.c, 1987):  initial fine mesh upper corner = 28.9095 74.162 70.372
NOsh_setupCalcMGAUTO (/home/ubuntu/git/apbs-pdb2pqr/apbs/src/generic/nosh.c, 1992):  initial fine mesh lower corner = -41.5015 8.401 1.389
NOsh_setupCalcMGAUTO (/home/ubuntu/git/apbs-pdb2pqr/apbs/src/generic/nosh.c, 2053):  final fine mesh upper corner = 28.9095 74.162 70.372
NOsh_setupCalcMGAUTO (/home/ubuntu/git/apbs-pdb2pqr/apbs/src/generic/nosh.c, 2058):  final fine mesh lower corner = -41.5015 8.401 1.389
NOsh_setupMGAUTO:  Resetting boundary flags
NOsh_setupCalc:  Mapping ELEC statement 0 (1) to calculation 1 (2)
Vnm_tstart: starting timer 27 (Setup timer)..
Setting up PBE object...
Vpbe_ctor2:  solute radius = 32.021
Vpbe_ctor2:  solute dimensions = 50.411 x 45.761 x 48.983
Vpbe_ctor2:  solute charge = -3
Vpbe_ctor2:  bulk ionic strength = 0
Vpbe_ctor2:  xkappa = 0
Vpbe_ctor2:  Debye length = 0
Vpbe_ctor2:  zkappa2 = 0
Vpbe_ctor2:  zmagic = 7042.98
Vpbe_ctor2:  Constructing Vclist with 75 x 75 x 75 table
Vclist_ctor2:  Using 75 x 75 x 75 hash table
Vclist_ctor2:  automatic domain setup.
Vclist_ctor2:  Using 1.9 max radius
Vclist_setupGrid:  Grid lengths = (58.874, 54.215, 57.493)
Vclist_setupGrid:  Grid lower corner = (-35.733, 14.174, 7.134)
Vclist_assignAtoms:  Have 2934611 atom entries
Vacc_storeParms:  Surf. density = 10
Vacc_storeParms:  Max area = 191.134
Vacc_storeParms:  Using 1936-point reference sphere
Setting up PDE object...
Vpmp_ctor2:  Using meth = 2, mgsolv = 1
Setting PDE center to local center...
Vpmg_fillco:  filling in source term.
fillcoCharge:  Calling fillcoChargeSpline2...
Vpmg_fillco:  filling in source term.
Vpmg_fillco:  marking ion and solvent accessibility.
fillcoCoef:  Calling fillcoCoefMol...
Vacc_SASA: Time elapsed: 0.181111
Vpmg_fillco:  done filling coefficient arrays
Vpmg_fillco:  filling boundary arrays
Vpmg_fillco:  done filling boundary arrays
Vnm_tstop: stopping timer 27 (Setup timer).  CPU TIME = 5.091760e-01
Vnm_tstart: starting timer 28 (Solver timer)..
Vnm_tstart: starting timer 30 (Vmgdrv2: fine problem setup)..
Vbuildops: Fine: (129, 129, 129)
Vbuildops: Operator stencil (lev, numdia) = (1, 4)
Vnm_tstop: stopping timer 30 (Vmgdrv2: fine problem setup).  CPU TIME = 4.008100e-02
Vnm_tstart: starting timer 30 (Vmgdrv2: coarse problem setup)..
Vbuildops: Galer: (065, 065, 065)
Vbuildops: Galer: (033, 033, 033)
Vbuildops: Galer: (017, 017, 017)
Vbuildops: Galer: (009, 009, 009)
Vbuildops: Galer: (005, 005, 005)
Vnm_tstop: stopping timer 30 (Vmgdrv2: coarse problem setup).  CPU TIME = 1.240170e-01
Vnm_tstart: starting timer 30 (Vmgdrv2: solve)..
Vnm_tstop: stopping timer 40 (MG iteration).  CPU TIME = 7.236180e-01
Vprtstp: iteration = 0
Vprtstp: relative residual = 1.000000e+00
Vprtstp: contraction number = 1.000000e+00
Vprtstp: iteration = 1
Vprtstp: relative residual = 1.104771e-01
Vprtstp: contraction number = 1.104771e-01
Vprtstp: iteration = 2
Vprtstp: relative residual = 1.407814e-02
Vprtstp: contraction number = 1.274304e-01
Vprtstp: iteration = 3
Vprtstp: relative residual = 1.995673e-03
Vprtstp: contraction number = 1.417569e-01
Vprtstp: iteration = 4
Vprtstp: relative residual = 3.073279e-04
Vprtstp: contraction number = 1.539971e-01
Vprtstp: iteration = 5
Vprtstp: relative residual = 5.017704e-05
Vprtstp: contraction number = 1.632688e-01
Vprtstp: iteration = 6
Vprtstp: relative residual = 8.368740e-06
Vprtstp: contraction number = 1.667842e-01
Vprtstp: iteration = 7
Vprtstp: relative residual = 1.437967e-06
Vprtstp: contraction number = 1.718260e-01
Vprtstp: iteration = 8
Vprtstp: relative residual = 2.720925e-07
Vprtstp: contraction number = 1.892203e-01
Vnm_tstop: stopping timer 30 (Vmgdrv2: solve).  CPU TIME = 1.014643e+00
Vnm_tstop: stopping timer 28 (Solver timer).  CPU TIME = 1.202084e+00
Vpmg_setPart:  lower corner = (-49.1454, 2.38465, -5.75505)
Vpmg_setPart:  upper corner = (36.5534, 80.1783, 77.5161)
Vpmg_setPart:  actual minima = (-49.1454, 2.38465, -5.75505)
Vpmg_setPart:  actual maxima = (36.5534, 80.1783, 77.5161)
Vpmg_setPart:  bflag[FRONT] = 0
Vpmg_setPart:  bflag[BACK] = 0
Vpmg_setPart:  bflag[LEFT] = 0
Vpmg_setPart:  bflag[RIGHT] = 0
Vpmg_setPart:  bflag[UP] = 0
Vpmg_setPart:  bflag[DOWN] = 0
Vnm_tstart: starting timer 29 (Energy timer)..
Vpmg_energy:  calculating only q-phi energy
Vpmg_qfEnergyVolume:  Calculating energy
Vpmg_energy:  qfEnergy = 6.279900734989E+04 kT
Vnm_tstop: stopping timer 29 (Energy timer).  CPU TIME = 3.420000e-03
Vnm_tstart: starting timer 30 (Force timer)..
Vnm_tstop: stopping timer 30 (Force timer).  CPU TIME = 1.000000e-06
Vnm_tstart: starting timer 27 (Setup timer)..
Setting up PBE object...
Vpbe_ctor2:  solute radius = 32.021
Vpbe_ctor2:  solute dimensions = 50.411 x 45.761 x 48.983
Vpbe_ctor2:  solute charge = -3
Vpbe_ctor2:  bulk ionic strength = 0
Vpbe_ctor2:  xkappa = 0
Vpbe_ctor2:  Debye length = 0
Vpbe_ctor2:  zkappa2 = 0
Vpbe_ctor2:  zmagic = 7042.98
Vpbe_ctor2:  Constructing Vclist with 75 x 75 x 75 table
Vclist_ctor2:  Using 75 x 75 x 75 hash table
Vclist_ctor2:  automatic domain setup.
Vclist_ctor2:  Using 1.9 max radius
Vclist_setupGrid:  Grid lengths = (58.874, 54.215, 57.493)
Vclist_setupGrid:  Grid lower corner = (-35.733, 14.174, 7.134)
Vclist_assignAtoms:  Have 2934611 atom entries
Vacc_storeParms:  Surf. density = 10
Vacc_storeParms:  Max area = 191.134
Vacc_storeParms:  Using 1936-point reference sphere
Setting up PDE object...
Vpmp_ctor2:  Using meth = 2, mgsolv = 1
Setting PDE center to local center...
Vpmg_ctor2:  Filling boundary with old solution!
VPMG::focusFillBound -- New mesh mins = -41.5015, 8.401, 1.389
VPMG::focusFillBound -- New mesh maxs = 28.9095, 74.162, 70.372
VPMG::focusFillBound -- Old mesh mins = -49.1454, 2.38465, -5.75505
VPMG::focusFillBound -- Old mesh maxs = 36.5534, 80.1783, 77.5161
VPMG::extEnergy:  energy flag = 1
Vpmg_setPart:  lower corner = (-41.5015, 8.401, 1.389)
Vpmg_setPart:  upper corner = (28.9095, 74.162, 70.372)
Vpmg_setPart:  actual minima = (-49.1454, 2.38465, -5.75505)
Vpmg_setPart:  actual maxima = (36.5534, 80.1783, 77.5161)
Vpmg_setPart:  bflag[FRONT] = 0
Vpmg_setPart:  bflag[BACK] = 0
Vpmg_setPart:  bflag[LEFT] = 0
Vpmg_setPart:  bflag[RIGHT] = 0
Vpmg_setPart:  bflag[UP] = 0
Vpmg_setPart:  bflag[DOWN] = 0
VPMG::extEnergy:   Finding extEnergy dimensions...
VPMG::extEnergy    Disj part lower corner = (-41.5015, 8.401, 1.389)
VPMG::extEnergy    Disj part upper corner = (28.9095, 74.162, 70.372)
VPMG::extEnergy    Old lower corner = (-49.1454, 2.38465, -5.75505)
VPMG::extEnergy    Old upper corner = (36.5534, 80.1783, 77.5161)
Vpmg_qmEnergy:  Zero energy for zero ionic strength!
VPMG::extEnergy: extQmEnergy = 0 kT
Vpmg_qfEnergyVolume:  Calculating energy
VPMG::extEnergy: extQfEnergy = 0 kT
VPMG::extEnergy: extDiEnergy = 0.510691 kT
Vpmg_fillco:  filling in source term.
fillcoCharge:  Calling fillcoChargeSpline2...
Vpmg_fillco:  filling in source term.
Vpmg_fillco:  marking ion and solvent accessibility.
fillcoCoef:  Calling fillcoCoefMol...
Vacc_SASA: Time elapsed: 0.179150
Vpmg_fillco:  done filling coefficient arrays
Vnm_tstop: stopping timer 27 (Setup timer).  CPU TIME = 6.234220e-01
Vnm_tstart: starting timer 28 (Solver timer)..
Vnm_tstart: starting timer 30 (Vmgdrv2: fine problem setup)..
Vbuildops: Fine: (129, 129, 129)
Vbuildops: Operator stencil (lev, numdia) = (1, 4)
Vnm_tstop: stopping timer 30 (Vmgdrv2: fine problem setup).  CPU TIME = 3.859000e-02
Vnm_tstart: starting timer 30 (Vmgdrv2: coarse problem setup)..
Vbuildops: Galer: (065, 065, 065)
Vbuildops: Galer: (033, 033, 033)
Vbuildops: Galer: (017, 017, 017)
Vbuildops: Galer: (009, 009, 009)
Vbuildops: Galer: (005, 005, 005)
Vnm_tstop: stopping timer 30 (Vmgdrv2: coarse problem setup).  CPU TIME = 1.215960e-01
Vnm_tstart: starting timer 30 (Vmgdrv2: solve)..
Vnm_tstop: stopping timer 40 (MG iteration).  CPU TIME = 2.559948e+00
Vprtstp: iteration = 0
Vprtstp: relative residual = 1.000000e+00
Vprtstp: contraction number = 1.000000e+00
Vprtstp: iteration = 1
Vprtstp: relative residual = 1.287377e-01
Vprtstp: contraction number = 1.287377e-01
Vprtstp: iteration = 2
Vprtstp: relative residual = 1.679246e-02
Vprtstp: contraction number = 1.304394e-01
Vprtstp: iteration = 3
Vprtstp: relative residual = 2.421199e-03
Vprtstp: contraction number = 1.441837e-01
Vprtstp: iteration = 4
Vprtstp: relative residual = 3.792913e-04
Vprtstp: contraction number = 1.566544e-01
Vprtstp: iteration = 5
Vprtstp: relative residual = 6.273073e-05
Vprtstp: contraction number = 1.653893e-01
Vprtstp: iteration = 6
Vprtstp: relative residual = 1.050571e-05
Vprtstp: contraction number = 1.674731e-01
Vprtstp: iteration = 7
Vprtstp: relative residual = 1.783957e-06
Vprtstp: contraction number = 1.698084e-01
Vprtstp: iteration = 8
Vprtstp: relative residual = 3.508544e-07
Vprtstp: contraction number = 1.966720e-01
Vnm_tstop: stopping timer 30 (Vmgdrv2: solve).  CPU TIME = 1.017669e+00
Vnm_tstop: stopping timer 28 (Solver timer).  CPU TIME = 1.200394e+00
Vpmg_setPart:  lower corner = (-41.5015, 8.401, 1.389)
Vpmg_setPart:  upper corner = (28.9095, 74.162, 70.372)
Vpmg_setPart:  actual minima = (-41.5015, 8.401, 1.389)
Vpmg_setPart:  actual maxima = (28.9095, 74.162, 70.372)
Vpmg_setPart:  bflag[FRONT] = 0
Vpmg_setPart:  bflag[BACK] = 0
Vpmg_setPart:  bflag[LEFT] = 0
Vpmg_setPart:  bflag[RIGHT] = 0
Vpmg_setPart:  bflag[UP] = 0
Vpmg_setPart:  bflag[DOWN] = 0
Vnm_tstart: starting timer 29 (Energy timer)..
Vpmg_energy:  calculating only q-phi energy
Vpmg_qfEnergyVolume:  Calculating energy
Vpmg_energy:  qfEnergy = 9.010317174362E+04 kT
Vnm_tstop: stopping timer 29 (Energy timer).  CPU TIME = 3.402000e-03
Vnm_tstart: starting timer 30 (Force timer)..
Vnm_tstop: stopping timer 30 (Force timer).  CPU TIME = 2.000000e-06
Vgrid_writeDX:  Opening virtual socket...
Vgrid_writeDX:  Writing to virtual socket...
Vgrid_writeDX:  Writing comments for ASC format.
printEnergy:  Performing global reduction (sum)
Vcom_reduce:  Not compiled with MPI, doing simple copy.
Vnm_tstop: stopping timer 26 (APBS WALL CLOCK).  CPU TIME = 4.216333e+00
##############################################################################
# MC-shell I/O capture file.
# Creation Date and Time:  Mon Jul  7 16:08:30 2025

##############################################################################
Vgrid_readDX:  Grid dimensions 129 x 129 x 129 grid
Vgrid_readDX:  Grid origin = (-41.5015, 8.401, 1.389)
Vgrid_readDX:  Grid spacings = (0.550086, 0.513758, 0.53893)
Vgrid_readDX:  allocating 129 x 129 x 129 doubles for storage
##############################################################################
# MC-shell I/O capture file.
# Creation Date and Time:  Mon Jul  7 16:30:52 2025

##############################################################################
Hello world from PE 0
Vnm_tstart: starting timer 26 (APBS WALL CLOCK)..
NOsh_parseInput:  Starting file parsing...
NOsh: Parsing READ section
NOsh: Storing molecule 0 path complex_A
NOsh: Done parsing READ section
NOsh: Done parsing READ section (nmol=1, ndiel=0, nkappa=0, ncharge=0, npot=0)
NOsh: Parsing ELEC section
NOsh_parseMG: Parsing parameters for MG calculation
NOsh_parseMG:  Parsing dime...
PBEparm_parseToken:  trying dime...
MGparm_parseToken:  trying dime...
NOsh_parseMG:  Parsing cglen...
PBEparm_parseToken:  trying cglen...
MGparm_parseToken:  trying cglen...
NOsh_parseMG:  Parsing fglen...
PBEparm_parseToken:  trying fglen...
MGparm_parseToken:  trying fglen...
NOsh_parseMG:  Parsing cgcent...
PBEparm_parseToken:  trying cgcent...
MGparm_parseToken:  trying cgcent...
NOsh_parseMG:  Parsing fgcent...
PBEparm_parseToken:  trying fgcent...
MGparm_parseToken:  trying fgcent...
NOsh_parseMG:  Parsing mol...
PBEparm_parseToken:  trying mol...
NOsh_parseMG:  Parsing lpbe...
PBEparm_parseToken:  trying lpbe...
NOsh: parsed lpbe
NOsh_parseMG:  Parsing bcfl...
PBEparm_parseToken:  trying bcfl...
NOsh_parseMG:  Parsing pdie...
PBEparm_parseToken:  trying pdie...
NOsh_parseMG:  Parsing sdie...
PBEparm_parseToken:  trying sdie...
NOsh_parseMG:  Parsing srfm...
PBEparm_parseToken:  trying srfm...
NOsh_parseMG:  Parsing chgm...
PBEparm_parseToken:  trying chgm...
MGparm_parseToken:  trying chgm...
NOsh_parseMG:  Parsing sdens...
PBEparm_parseToken:  trying sdens...
NOsh_parseMG:  Parsing srad...
PBEparm_parseToken:  trying srad...
NOsh_parseMG:  Parsing swin...
PBEparm_parseToken:  trying swin...
NOsh_parseMG:  Parsing temp...
PBEparm_parseToken:  trying temp...
NOsh_parseMG:  Parsing calcenergy...
PBEparm_parseToken:  trying calcenergy...
NOsh_parseMG:  Parsing calcforce...
PBEparm_parseToken:  trying calcforce...
NOsh_parseMG:  Parsing write...
PBEparm_parseToken:  trying write...
NOsh_parseMG:  Parsing end...
MGparm_check:  checking MGparm object of type 1.
NOsh:  nlev = 6, dime = (129, 129, 129)
NOsh: Done parsing ELEC section (nelec = 1)
NOsh: Parsing PRINT section
NOsh: Done parsing PRINT section
NOsh: Done parsing PRINT section
NOsh: Done parsing file (got QUIT)
Valist_readPQR: Counted 2983 atoms
Valist_getStatistics:  Max atom coordinate:  (17.603, 62.851, 59.089)
Valist_getStatistics:  Min atom coordinate:  (-30.195, 19.712, 12.672)
Valist_getStatistics:  Molecule center:  (-6.296, 41.2815, 35.8805)
NOsh_setupCalcMGAUTO(/home/ubuntu/git/apbs-pdb2pqr/apbs/src/generic/nosh.c, 1855):  coarse grid center = -6.296 41.2815 35.8805
NOsh_setupCalcMGAUTO(/home/ubuntu/git/apbs-pdb2pqr/apbs/src/generic/nosh.c, 1860):  fine grid center = -6.296 41.2815 35.8805
NOsh_setupCalcMGAUTO (/home/ubuntu/git/apbs-pdb2pqr/apbs/src/generic/nosh.c, 1872):  Coarse grid spacing = 0.669521, 0.607763, 0.650555
NOsh_setupCalcMGAUTO (/home/ubuntu/git/apbs-pdb2pqr/apbs/src/generic/nosh.c, 1874):  Fine grid spacing = 0.550086, 0.513758, 0.53893
NOsh_setupCalcMGAUTO (/home/ubuntu/git/apbs-pdb2pqr/apbs/src/generic/nosh.c, 1876):  Displacement between fine and coarse grids = 0, 0, 0
NOsh:  2 levels of focusing with 0.821611, 0.845326, 0.828415 reductions
NOsh_setupMGAUTO:  Resetting boundary flags
NOsh_setupCalcMGAUTO (/home/ubuntu/git/apbs-pdb2pqr/apbs/src/generic/nosh.c, 1970):  starting mesh repositioning.
NOsh_setupCalcMGAUTO (/home/ubuntu/git/apbs-pdb2pqr/apbs/src/generic/nosh.c, 1972):  coarse mesh center = -6.296 41.2815 35.8805
NOsh_setupCalcMGAUTO (/home/ubuntu/git/apbs-pdb2pqr/apbs/src/generic/nosh.c, 1977):  coarse mesh upper corner = 36.5534 80.1783 77.5161
NOsh_setupCalcMGAUTO (/home/ubuntu/git/apbs-pdb2pqr/apbs/src/generic/nosh.c, 1982):  coarse mesh lower corner = -49.1454 2.38465 -5.75505
NOsh_setupCalcMGAUTO (/home/ubuntu/git/apbs-pdb2pqr/apbs/src/generic/nosh.c, 1987):  initial fine mesh upper corner = 28.9095 74.162 70.372
NOsh_setupCalcMGAUTO (/home/ubuntu/git/apbs-pdb2pqr/apbs/src/generic/nosh.c, 1992):  initial fine mesh lower corner = -41.5015 8.401 1.389
NOsh_setupCalcMGAUTO (/home/ubuntu/git/apbs-pdb2pqr/apbs/src/generic/nosh.c, 2053):  final fine mesh upper corner = 28.9095 74.162 70.372
NOsh_setupCalcMGAUTO (/home/ubuntu/git/apbs-pdb2pqr/apbs/src/generic/nosh.c, 2058):  final fine mesh lower corner = -41.5015 8.401 1.389
NOsh_setupMGAUTO:  Resetting boundary flags
NOsh_setupCalc:  Mapping ELEC statement 0 (1) to calculation 1 (2)
Vnm_tstart: starting timer 27 (Setup timer)..
Setting up PBE object...
Vpbe_ctor2:  solute radius = 32.021
Vpbe_ctor2:  solute dimensions = 50.411 x 45.761 x 48.983
Vpbe_ctor2:  solute charge = -3
Vpbe_ctor2:  bulk ionic strength = 0
Vpbe_ctor2:  xkappa = 0
Vpbe_ctor2:  Debye length = 0
Vpbe_ctor2:  zkappa2 = 0
Vpbe_ctor2:  zmagic = 7042.98
Vpbe_ctor2:  Constructing Vclist with 75 x 75 x 75 table
Vclist_ctor2:  Using 75 x 75 x 75 hash table
Vclist_ctor2:  automatic domain setup.
Vclist_ctor2:  Using 1.9 max radius
Vclist_setupGrid:  Grid lengths = (58.874, 54.215, 57.493)
Vclist_setupGrid:  Grid lower corner = (-35.733, 14.174, 7.134)
Vclist_assignAtoms:  Have 2934611 atom entries
Vacc_storeParms:  Surf. density = 10
Vacc_storeParms:  Max area = 191.134
Vacc_storeParms:  Using 1936-point reference sphere
Setting up PDE object...
Vpmp_ctor2:  Using meth = 2, mgsolv = 1
Setting PDE center to local center...
Vpmg_fillco:  filling in source term.
fillcoCharge:  Calling fillcoChargeSpline2...
Vpmg_fillco:  filling in source term.
Vpmg_fillco:  marking ion and solvent accessibility.
fillcoCoef:  Calling fillcoCoefMol...
Vacc_SASA: Time elapsed: 0.180904
Vpmg_fillco:  done filling coefficient arrays
Vpmg_fillco:  filling boundary arrays
Vpmg_fillco:  done filling boundary arrays
Vnm_tstop: stopping timer 27 (Setup timer).  CPU TIME = 5.071330e-01
Vnm_tstart: starting timer 28 (Solver timer)..
Vnm_tstart: starting timer 30 (Vmgdrv2: fine problem setup)..
Vbuildops: Fine: (129, 129, 129)
Vbuildops: Operator stencil (lev, numdia) = (1, 4)
Vnm_tstop: stopping timer 30 (Vmgdrv2: fine problem setup).  CPU TIME = 3.960100e-02
Vnm_tstart: starting timer 30 (Vmgdrv2: coarse problem setup)..
Vbuildops: Galer: (065, 065, 065)
Vbuildops: Galer: (033, 033, 033)
Vbuildops: Galer: (017, 017, 017)
Vbuildops: Galer: (009, 009, 009)
Vbuildops: Galer: (005, 005, 005)
Vnm_tstop: stopping timer 30 (Vmgdrv2: coarse problem setup).  CPU TIME = 1.239870e-01
Vnm_tstart: starting timer 30 (Vmgdrv2: solve)..
Vnm_tstop: stopping timer 40 (MG iteration).  CPU TIME = 7.224990e-01
Vprtstp: iteration = 0
Vprtstp: relative residual = 1.000000e+00
Vprtstp: contraction number = 1.000000e+00
Vprtstp: iteration = 1
Vprtstp: relative residual = 1.104771e-01
Vprtstp: contraction number = 1.104771e-01
Vprtstp: iteration = 2
Vprtstp: relative residual = 1.407814e-02
Vprtstp: contraction number = 1.274304e-01
Vprtstp: iteration = 3
Vprtstp: relative residual = 1.995673e-03
Vprtstp: contraction number = 1.417569e-01
Vprtstp: iteration = 4
Vprtstp: relative residual = 3.073279e-04
Vprtstp: contraction number = 1.539971e-01
Vprtstp: iteration = 5
Vprtstp: relative residual = 5.017704e-05
Vprtstp: contraction number = 1.632688e-01
Vprtstp: iteration = 6
Vprtstp: relative residual = 8.368740e-06
Vprtstp: contraction number = 1.667842e-01
Vprtstp: iteration = 7
Vprtstp: relative residual = 1.437967e-06
Vprtstp: contraction number = 1.718260e-01
Vprtstp: iteration = 8
Vprtstp: relative residual = 2.720925e-07
Vprtstp: contraction number = 1.892203e-01
Vnm_tstop: stopping timer 30 (Vmgdrv2: solve).  CPU TIME = 1.060541e+00
Vnm_tstop: stopping timer 28 (Solver timer).  CPU TIME = 1.247610e+00
Vpmg_setPart:  lower corner = (-49.1454, 2.38465, -5.75505)
Vpmg_setPart:  upper corner = (36.5534, 80.1783, 77.5161)
Vpmg_setPart:  actual minima = (-49.1454, 2.38465, -5.75505)
Vpmg_setPart:  actual maxima = (36.5534, 80.1783, 77.5161)
Vpmg_setPart:  bflag[FRONT] = 0
Vpmg_setPart:  bflag[BACK] = 0
Vpmg_setPart:  bflag[LEFT] = 0
Vpmg_setPart:  bflag[RIGHT] = 0
Vpmg_setPart:  bflag[UP] = 0
Vpmg_setPart:  bflag[DOWN] = 0
Vnm_tstart: starting timer 29 (Energy timer)..
Vpmg_energy:  calculating only q-phi energy
Vpmg_qfEnergyVolume:  Calculating energy
Vpmg_energy:  qfEnergy = 6.279900734989E+04 kT
Vnm_tstop: stopping timer 29 (Energy timer).  CPU TIME = 3.411000e-03
Vnm_tstart: starting timer 30 (Force timer)..
Vnm_tstop: stopping timer 30 (Force timer).  CPU TIME = 0.000000e+00
Vnm_tstart: starting timer 27 (Setup timer)..
Setting up PBE object...
Vpbe_ctor2:  solute radius = 32.021
Vpbe_ctor2:  solute dimensions = 50.411 x 45.761 x 48.983
Vpbe_ctor2:  solute charge = -3
Vpbe_ctor2:  bulk ionic strength = 0
Vpbe_ctor2:  xkappa = 0
Vpbe_ctor2:  Debye length = 0
Vpbe_ctor2:  zkappa2 = 0
Vpbe_ctor2:  zmagic = 7042.98
Vpbe_ctor2:  Constructing Vclist with 75 x 75 x 75 table
Vclist_ctor2:  Using 75 x 75 x 75 hash table
Vclist_ctor2:  automatic domain setup.
Vclist_ctor2:  Using 1.9 max radius
Vclist_setupGrid:  Grid lengths = (58.874, 54.215, 57.493)
Vclist_setupGrid:  Grid lower corner = (-35.733, 14.174, 7.134)
Vclist_assignAtoms:  Have 2934611 atom entries
Vacc_storeParms:  Surf. density = 10
Vacc_storeParms:  Max area = 191.134
Vacc_storeParms:  Using 1936-point reference sphere
Setting up PDE object...
Vpmp_ctor2:  Using meth = 2, mgsolv = 1
Setting PDE center to local center...
Vpmg_ctor2:  Filling boundary with old solution!
VPMG::focusFillBound -- New mesh mins = -41.5015, 8.401, 1.389
VPMG::focusFillBound -- New mesh maxs = 28.9095, 74.162, 70.372
VPMG::focusFillBound -- Old mesh mins = -49.1454, 2.38465, -5.75505
VPMG::focusFillBound -- Old mesh maxs = 36.5534, 80.1783, 77.5161
VPMG::extEnergy:  energy flag = 1
Vpmg_setPart:  lower corner = (-41.5015, 8.401, 1.389)
Vpmg_setPart:  upper corner = (28.9095, 74.162, 70.372)
Vpmg_setPart:  actual minima = (-49.1454, 2.38465, -5.75505)
Vpmg_setPart:  actual maxima = (36.5534, 80.1783, 77.5161)
Vpmg_setPart:  bflag[FRONT] = 0
Vpmg_setPart:  bflag[BACK] = 0
Vpmg_setPart:  bflag[LEFT] = 0
Vpmg_setPart:  bflag[RIGHT] = 0
Vpmg_setPart:  bflag[UP] = 0
Vpmg_setPart:  bflag[DOWN] = 0
VPMG::extEnergy:   Finding extEnergy dimensions...
VPMG::extEnergy    Disj part lower corner = (-41.5015, 8.401, 1.389)
VPMG::extEnergy    Disj part upper corner = (28.9095, 74.162, 70.372)
VPMG::extEnergy    Old lower corner = (-49.1454, 2.38465, -5.75505)
VPMG::extEnergy    Old upper corner = (36.5534, 80.1783, 77.5161)
Vpmg_qmEnergy:  Zero energy for zero ionic strength!
VPMG::extEnergy: extQmEnergy = 0 kT
Vpmg_qfEnergyVolume:  Calculating energy
VPMG::extEnergy: extQfEnergy = 0 kT
VPMG::extEnergy: extDiEnergy = 0.510691 kT
Vpmg_fillco:  filling in source term.
fillcoCharge:  Calling fillcoChargeSpline2...
Vpmg_fillco:  filling in source term.
Vpmg_fillco:  marking ion and solvent accessibility.
fillcoCoef:  Calling fillcoCoefMol...
Vacc_SASA: Time elapsed: 0.179445
Vpmg_fillco:  done filling coefficient arrays
Vnm_tstop: stopping timer 27 (Setup timer).  CPU TIME = 6.235280e-01
Vnm_tstart: starting timer 28 (Solver timer)..
Vnm_tstart: starting timer 30 (Vmgdrv2: fine problem setup)..
Vbuildops: Fine: (129, 129, 129)
Vbuildops: Operator stencil (lev, numdia) = (1, 4)
Vnm_tstop: stopping timer 30 (Vmgdrv2: fine problem setup).  CPU TIME = 3.883800e-02
Vnm_tstart: starting timer 30 (Vmgdrv2: coarse problem setup)..
Vbuildops: Galer: (065, 065, 065)
Vbuildops: Galer: (033, 033, 033)
Vbuildops: Galer: (017, 017, 017)
Vbuildops: Galer: (009, 009, 009)
Vbuildops: Galer: (005, 005, 005)
Vnm_tstop: stopping timer 30 (Vmgdrv2: coarse problem setup).  CPU TIME = 1.222310e-01
Vnm_tstart: starting timer 30 (Vmgdrv2: solve)..
Vnm_tstop: stopping timer 40 (MG iteration).  CPU TIME = 2.605720e+00
Vprtstp: iteration = 0
Vprtstp: relative residual = 1.000000e+00
Vprtstp: contraction number = 1.000000e+00
Vprtstp: iteration = 1
Vprtstp: relative residual = 1.287377e-01
Vprtstp: contraction number = 1.287377e-01
Vprtstp: iteration = 2
Vprtstp: relative residual = 1.679246e-02
Vprtstp: contraction number = 1.304394e-01
Vprtstp: iteration = 3
Vprtstp: relative residual = 2.421199e-03
Vprtstp: contraction number = 1.441837e-01
Vprtstp: iteration = 4
Vprtstp: relative residual = 3.792913e-04
Vprtstp: contraction number = 1.566544e-01
Vprtstp: iteration = 5
Vprtstp: relative residual = 6.273073e-05
Vprtstp: contraction number = 1.653893e-01
Vprtstp: iteration = 6
Vprtstp: relative residual = 1.050571e-05
Vprtstp: contraction number = 1.674731e-01
Vprtstp: iteration = 7
Vprtstp: relative residual = 1.783957e-06
Vprtstp: contraction number = 1.698084e-01
Vprtstp: iteration = 8
Vprtstp: relative residual = 3.508544e-07
Vprtstp: contraction number = 1.966720e-01
Vnm_tstop: stopping timer 30 (Vmgdrv2: solve).  CPU TIME = 1.057139e+00
Vnm_tstop: stopping timer 28 (Solver timer).  CPU TIME = 1.240479e+00
Vpmg_setPart:  lower corner = (-41.5015, 8.401, 1.389)
Vpmg_setPart:  upper corner = (28.9095, 74.162, 70.372)
Vpmg_setPart:  actual minima = (-41.5015, 8.401, 1.389)
Vpmg_setPart:  actual maxima = (28.9095, 74.162, 70.372)
Vpmg_setPart:  bflag[FRONT] = 0
Vpmg_setPart:  bflag[BACK] = 0
Vpmg_setPart:  bflag[LEFT] = 0
Vpmg_setPart:  bflag[RIGHT] = 0
Vpmg_setPart:  bflag[UP] = 0
Vpmg_setPart:  bflag[DOWN] = 0
Vnm_tstart: starting timer 29 (Energy timer)..
Vpmg_energy:  calculating only q-phi energy
Vpmg_qfEnergyVolume:  Calculating energy
Vpmg_energy:  qfEnergy = 9.010317174362E+04 kT
Vnm_tstop: stopping timer 29 (Energy timer).  CPU TIME = 3.388000e-03
Vnm_tstart: starting timer 30 (Force timer)..
Vnm_tstop: stopping timer 30 (Force timer).  CPU TIME = 1.000000e-06
Vgrid_writeDX:  Opening virtual socket...
Vgrid_writeDX:  Writing to virtual socket...
Vgrid_writeDX:  Writing comments for ASC format.
printEnergy:  Performing global reduction (sum)
Vcom_reduce:  Not compiled with MPI, doing simple copy.
Vnm_tstop: stopping timer 26 (APBS WALL CLOCK).  CPU TIME = 4.304412e+00
##############################################################################
# MC-shell I/O capture file.
# Creation Date and Time:  Mon Jul  7 16:30:56 2025

##############################################################################
Vgrid_readDX:  Grid dimensions 129 x 129 x 129 grid
Vgrid_readDX:  Grid origin = (-41.5015, 8.401, 1.389)
Vgrid_readDX:  Grid spacings = (0.550086, 0.513758, 0.53893)
Vgrid_readDX:  allocating 129 x 129 x 129 doubles for storage
