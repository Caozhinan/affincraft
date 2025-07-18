"""
extractPDB.py: Extract selected chains from a PDB and save the extracted chains to an output file. 
Pablo Gainza - LPDI STI EPFL 2019
Released under an Apache License 2.0
"""
from Bio.PDB import *

from Bio.SeqUtils import IUPACData
PROTEIN_LETTERS = [x.upper() for x in IUPACData.protein_letters_3to1.keys()]

# Exclude disordered atoms.
class NotDisordered(Select):  
    def accept_atom(self, atom):  
        # 对配体原子更宽松的处理  
        if atom.get_parent().get_resname() == "UNK":  
            return not atom.is_disordered() or atom.get_altloc() in ["A", "1", ""]  
        return not atom.is_disordered() or atom.get_altloc() == "A" or atom.get_altloc() == "1"


def find_modified_amino_acids(path):
    """
    Contributed by github user jomimc - find modified amino acids in the PDB (e.g. MSE)
    """
    res_set = set()
    for line in open(path, 'r'):
        if line[:6] == 'SEQRES':
            for res in line.split()[4:]:
                res_set.add(res)
    for res in list(res_set):
        if res in PROTEIN_LETTERS:
            res_set.remove(res)
    return res_set


def extractPDB(
    infilename, outfilename, chain_ids=None, ligand_code=None, ligand_chain=None
):
    # extract the chain_ids from infilename and save in outfilename. 
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure(infilename, infilename)
    model = Selection.unfold_entities(struct, "M")[0]
    chains = Selection.unfold_entities(struct, "C")
    # Select residues to extract and build new structure
    structBuild = StructureBuilder.StructureBuilder()
    structBuild.init_structure("output")
    structBuild.init_seg(" ")
    structBuild.init_model(0)
    outputStruct = structBuild.get_structure()

    # Load a list of non-standard amino acid names -- these are
    # typically listed under HETATM, so they would be typically
    # ignored by the orginal algorithm
    modified_amino_acids = find_modified_amino_acids(infilename)

    for chain in model:
        if (
            chain_ids == None
            or chain.get_id() in chain_ids
        ):
            structBuild.init_chain(chain.get_id())
            for residue in chain:
                het = residue.get_id()
                if het[0] == " ":
                    outputStruct[0][chain.get_id()].add(residue)
                elif het[0][-3:] in modified_amino_acids:
                    outputStruct[0][chain.get_id()].add(residue)
                elif ligand_code is not None and het[0][-3:] == ligand_code:
                    outputStruct[0][chain.get_id()].add(residue)

        # ligand might be in different chain
        elif ligand_code is not None and ligand_chain is not None and chain.get_id() == ligand_chain:
            structBuild.init_chain(chain.get_id())
            for residue in chain:
                het = residue.get_id()
                if het[0][-3:] == ligand_code:
                    outputStruct[0][chain.get_id()].add(residue)


    # Output the selected residues
    pdbio = PDBIO()
    pdbio.set_structure(outputStruct)
    pdbio.save(outfilename, select=NotDisordered())

