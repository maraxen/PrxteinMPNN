"""Unit tests for the prxteinmpnn.io.parsing submodule."""

import pathlib
import tempfile
from io import StringIO

import mdtraj as md
import numpy as np
import pytest
from biotite.structure import Atom, AtomArray, AtomArrayStack, array as strucarray
from chex import assert_trees_all_close

from prxteinmpnn.io.parsing import (
    _check_if_file_empty,
    af_to_mpnn,
    atom_array_dihedrals,
    atom_names_to_index,
    compute_cb_precise,
    extend_coordinate,
    mpnn_to_af,
    parse_input,
    protein_sequence_to_string,
    residue_names_to_aatype,
    string_key_to_index,
    string_to_protein_sequence,
)
from prxteinmpnn.utils.data_structures import ProteinTuple
from prxteinmpnn.utils.residue_constants import resname_to_idx, restype_order, unk_restype_index

PDB_STRING = """                                     
MODEL        1                                                                  
ATOM      1  N   GLY A   1      -6.778  -1.424   4.200  1.00  0.00           N  
ATOM      2  CA  GLY A   1      -6.878  -0.708   2.896  1.00  0.00           C  
ATOM      3  C   GLY A   1      -5.557  -0.840   2.138  1.00  0.00           C  
ATOM      4  O   GLY A   1      -4.640  -1.504   2.579  1.00  0.00           O  
ATOM      5  H1  GLY A   1      -5.778  -1.527   4.462  1.00  0.00           H  
ATOM      6  H2  GLY A   1      -7.214  -2.365   4.112  1.00  0.00           H  
ATOM      7  H3  GLY A   1      -7.273  -0.879   4.933  1.00  0.00           H  
ATOM      8  HA2 GLY A   1      -7.677  -1.140   2.309  1.00  0.00           H  
ATOM      9  HA3 GLY A   1      -7.085   0.336   3.073  1.00  0.00           H  
ATOM     10  N   TYR A   2      -5.452  -0.212   0.999  1.00  0.00           N  
ATOM     11  CA  TYR A   2      -4.189  -0.302   0.213  1.00  0.00           C  
ATOM     12  C   TYR A   2      -3.197   0.744   0.717  1.00  0.00           C  
ATOM     13  O   TYR A   2      -3.252   1.898   0.342  1.00  0.00           O  
ATOM     14  CB  TYR A   2      -4.490  -0.048  -1.264  1.00  0.00           C  
ATOM     15  CG  TYR A   2      -3.216  -0.141  -2.068  1.00  0.00           C  
ATOM     16  CD1 TYR A   2      -2.780  -1.383  -2.544  1.00  0.00           C  
ATOM     17  CD2 TYR A   2      -2.474   1.015  -2.342  1.00  0.00           C  
ATOM     18  CE1 TYR A   2      -1.601  -1.470  -3.293  1.00  0.00           C  
ATOM     19  CE2 TYR A   2      -1.295   0.927  -3.091  1.00  0.00           C  
ATOM     20  CZ  TYR A   2      -0.858  -0.315  -3.568  1.00  0.00           C  
ATOM     21  OH  TYR A   2       0.304  -0.402  -4.307  1.00  0.00           O  
ATOM     22  H   TYR A   2      -6.203   0.317   0.660  1.00  0.00           H  
ATOM     23  HA  TYR A   2      -3.763  -1.288   0.328  1.00  0.00           H  
ATOM     24  HB2 TYR A   2      -5.189  -0.788  -1.616  1.00  0.00           H  
ATOM     25  HB3 TYR A   2      -4.918   0.937  -1.380  1.00  0.00           H  
ATOM     26  HD1 TYR A   2      -3.352  -2.273  -2.332  1.00  0.00           H  
ATOM     27  HD2 TYR A   2      -2.811   1.973  -1.975  1.00  0.00           H  
ATOM     28  HE1 TYR A   2      -1.265  -2.429  -3.660  1.00  0.00           H  
ATOM     29  HE2 TYR A   2      -0.722   1.818  -3.303  1.00  0.00           H  
ATOM     30  HH  TYR A   2       0.765   0.437  -4.234  1.00  0.00           H  
ATOM     31  N   ASP A   3      -2.287   0.349   1.560  1.00  0.00           N  
ATOM     32  CA  ASP A   3      -1.287   1.320   2.084  1.00  0.00           C  
ATOM     33  C   ASP A   3      -0.301   1.677   0.964  1.00  0.00           C  
ATOM     34  O   ASP A   3       0.433   0.826   0.502  1.00  0.00           O  
ATOM     35  CB  ASP A   3      -0.523   0.687   3.249  1.00  0.00           C  
ATOM     36  CG  ASP A   3       0.418   1.724   3.866  1.00  0.00           C  
ATOM     37  OD1 ASP A   3      -0.069   2.762   4.283  1.00  0.00           O  
ATOM     38  OD2 ASP A   3       1.609   1.463   3.911  1.00  0.00           O  
ATOM     39  H   ASP A   3      -2.259  -0.588   1.846  1.00  0.00           H  
ATOM     40  HA  ASP A   3      -1.795   2.207   2.427  1.00  0.00           H  
ATOM     41  HB2 ASP A   3      -1.224   0.346   3.997  1.00  0.00           H  
ATOM     42  HB3 ASP A   3       0.055  -0.151   2.889  1.00  0.00           H  
ATOM     43  N   PRO A   4      -0.311   2.922   0.557  1.00  0.00           N  
ATOM     44  CA  PRO A   4       0.579   3.407  -0.513  1.00  0.00           C  
ATOM     45  C   PRO A   4       2.002   3.580   0.021  1.00  0.00           C  
ATOM     46  O   PRO A   4       2.943   3.738  -0.731  1.00  0.00           O  
ATOM     47  CB  PRO A   4      -0.033   4.752  -0.911  1.00  0.00           C  
ATOM     48  CG  PRO A   4      -0.876   5.221   0.298  1.00  0.00           C  
ATOM     49  CD  PRO A   4      -1.199   3.960   1.121  1.00  0.00           C  
ATOM     50  HA  PRO A   4       0.567   2.732  -1.353  1.00  0.00           H  
ATOM     51  HB2 PRO A   4       0.750   5.467  -1.123  1.00  0.00           H  
ATOM     52  HB3 PRO A   4      -0.670   4.630  -1.773  1.00  0.00           H  
ATOM     53  HG2 PRO A   4      -0.308   5.921   0.895  1.00  0.00           H  
ATOM     54  HG3 PRO A   4      -1.792   5.679  -0.042  1.00  0.00           H  
ATOM     55  HD2 PRO A   4      -0.977   4.125   2.167  1.00  0.00           H  
ATOM     56  HD3 PRO A   4      -2.232   3.677   0.991  1.00  0.00           H  
ATOM     57  N   GLU A   5       2.168   3.542   1.315  1.00  0.00           N  
ATOM     58  CA  GLU A   5       3.531   3.694   1.893  1.00  0.00           C  
ATOM     59  C   GLU A   5       4.420   2.567   1.368  1.00  0.00           C  
ATOM     60  O   GLU A   5       5.577   2.766   1.055  1.00  0.00           O  
ATOM     61  CB  GLU A   5       3.450   3.614   3.418  1.00  0.00           C  
ATOM     62  CG  GLU A   5       4.652   4.334   4.030  1.00  0.00           C  
ATOM     63  CD  GLU A   5       5.629   3.303   4.597  1.00  0.00           C  
ATOM     64  OE1 GLU A   5       5.649   2.193   4.090  1.00  0.00           O  
ATOM     65  OE2 GLU A   5       6.341   3.640   5.529  1.00  0.00           O  
ATOM     66  H   GLU A   5       1.397   3.406   1.904  1.00  0.00           H  
ATOM     67  HA  GLU A   5       3.944   4.648   1.600  1.00  0.00           H  
ATOM     68  HB2 GLU A   5       2.537   4.084   3.756  1.00  0.00           H  
ATOM     69  HB3 GLU A   5       3.457   2.580   3.727  1.00  0.00           H  
ATOM     70  HG2 GLU A   5       5.147   4.918   3.267  1.00  0.00           H  
ATOM     71  HG3 GLU A   5       4.318   4.985   4.823  1.00  0.00           H  
ATOM     72  N   THR A   6       3.881   1.383   1.261  1.00  0.00           N  
ATOM     73  CA  THR A   6       4.684   0.239   0.748  1.00  0.00           C  
ATOM     74  C   THR A   6       4.218  -0.101  -0.669  1.00  0.00           C  
ATOM     75  O   THR A   6       4.941  -0.687  -1.449  1.00  0.00           O  
ATOM     76  CB  THR A   6       4.486  -0.974   1.661  1.00  0.00           C  
ATOM     77  OG1 THR A   6       3.204  -1.540   1.423  1.00  0.00           O  
ATOM     78  CG2 THR A   6       4.589  -0.535   3.123  1.00  0.00           C  
ATOM     79  H   THR A   6       2.944   1.248   1.514  1.00  0.00           H  
ATOM     80  HA  THR A   6       5.729   0.510   0.729  1.00  0.00           H  
ATOM     81  HB  THR A   6       5.248  -1.709   1.456  1.00  0.00           H  
ATOM     82  HG1 THR A   6       3.211  -2.440   1.757  1.00  0.00           H  
ATOM     83 HG21 THR A   6       5.309   0.265   3.208  1.00  0.00           H  
ATOM     84 HG22 THR A   6       4.906  -1.371   3.728  1.00  0.00           H  
ATOM     85 HG23 THR A   6       3.625  -0.188   3.464  1.00  0.00           H  
ATOM     86  N   GLY A   7       3.014   0.272  -1.008  1.00  0.00           N  
ATOM     87  CA  GLY A   7       2.498  -0.018  -2.375  1.00  0.00           C  
ATOM     88  C   GLY A   7       1.780  -1.368  -2.385  1.00  0.00           C  
ATOM     89  O   GLY A   7       1.815  -2.091  -3.362  1.00  0.00           O  
ATOM     90  H   GLY A   7       2.450   0.749  -0.363  1.00  0.00           H  
ATOM     91  HA2 GLY A   7       1.808   0.760  -2.670  1.00  0.00           H  
ATOM     92  HA3 GLY A   7       3.323  -0.049  -3.071  1.00  0.00           H  
ATOM     93  N   THR A   8       1.127  -1.719  -1.310  1.00  0.00           N  
ATOM     94  CA  THR A   8       0.411  -3.025  -1.274  1.00  0.00           C  
ATOM     95  C   THR A   8      -0.641  -3.013  -0.162  1.00  0.00           C  
ATOM     96  O   THR A   8      -0.633  -2.164   0.707  1.00  0.00           O  
ATOM     97  CB  THR A   8       1.418  -4.147  -1.011  1.00  0.00           C  
ATOM     98  OG1 THR A   8       2.633  -3.587  -0.532  1.00  0.00           O  
ATOM     99  CG2 THR A   8       1.682  -4.912  -2.308  1.00  0.00           C  
ATOM    100  H   THR A   8       1.108  -1.125  -0.531  1.00  0.00           H  
ATOM    101  HA  THR A   8      -0.073  -3.195  -2.224  1.00  0.00           H  
ATOM    102  HB  THR A   8       1.018  -4.825  -0.273  1.00  0.00           H  
ATOM    103  HG1 THR A   8       3.275  -4.296  -0.450  1.00  0.00           H  
ATOM    104 HG21 THR A   8       0.813  -5.501  -2.562  1.00  0.00           H  
ATOM    105 HG22 THR A   8       2.533  -5.563  -2.176  1.00  0.00           H  
ATOM    106 HG23 THR A   8       1.885  -4.210  -3.104  1.00  0.00           H  
ATOM    107  N   TRP A   9      -1.545  -3.955  -0.184  1.00  0.00           N  
ATOM    108  CA  TRP A   9      -2.598  -4.011   0.868  1.00  0.00           C  
ATOM    109  C   TRP A   9      -1.960  -4.395   2.205  1.00  0.00           C  
ATOM    110  O   TRP A   9      -1.576  -5.528   2.417  1.00  0.00           O  
ATOM    111  CB  TRP A   9      -3.640  -5.067   0.487  1.00  0.00           C  
ATOM    112  CG  TRP A   9      -3.848  -5.068  -0.996  1.00  0.00           C  
ATOM    113  CD1 TRP A   9      -3.216  -5.882  -1.873  1.00  0.00           C  
ATOM    114  CD2 TRP A   9      -4.743  -4.236  -1.786  1.00  0.00           C  
ATOM    115  NE1 TRP A   9      -3.664  -5.596  -3.150  1.00  0.00           N  
ATOM    116  CE2 TRP A   9      -4.608  -4.591  -3.148  1.00  0.00           C  
ATOM    117  CE3 TRP A   9      -5.648  -3.218  -1.454  1.00  0.00           C  
ATOM    118  CZ2 TRP A   9      -5.349  -3.958  -4.146  1.00  0.00           C  
ATOM    119  CZ3 TRP A   9      -6.398  -2.577  -2.455  1.00  0.00           C  
ATOM    120  CH2 TRP A   9      -6.248  -2.947  -3.799  1.00  0.00           C  
ATOM    121  H   TRP A   9      -1.532  -4.627  -0.892  1.00  0.00           H  
ATOM    122  HA  TRP A   9      -3.075  -3.047   0.957  1.00  0.00           H  
ATOM    123  HB2 TRP A   9      -3.295  -6.041   0.800  1.00  0.00           H  
ATOM    124  HB3 TRP A   9      -4.575  -4.843   0.980  1.00  0.00           H  
ATOM    125  HD1 TRP A   9      -2.478  -6.628  -1.618  1.00  0.00           H  
ATOM    126  HE1 TRP A   9      -3.364  -6.042  -3.968  1.00  0.00           H  
ATOM    127  HE3 TRP A   9      -5.766  -2.927  -0.422  1.00  0.00           H  
ATOM    128  HZ2 TRP A   9      -5.228  -4.248  -5.180  1.00  0.00           H  
ATOM    129  HZ3 TRP A   9      -7.093  -1.796  -2.187  1.00  0.00           H  
ATOM    130  HH2 TRP A   9      -6.827  -2.451  -4.564  1.00  0.00           H  
ATOM    131  N   GLY B  10      -1.844  -3.461   3.109  1.00  0.00           N  
ATOM    132  CA  GLY B  10      -1.229  -3.774   4.431  1.00  0.00           C  
ATOM    133  C   GLY B  10      -1.877  -5.031   5.014  1.00  0.00           C  
ATOM    134  O   GLY B  10      -1.289  -6.092   4.882  1.00  0.00           O  
ATOM    135  OXT GLY B  10      -2.949  -4.912   5.584  1.00  0.00           O  
ATOM    136  H   GLY B  10      -2.159  -2.553   2.918  1.00  0.00           H  
ATOM    137  HA2 GLY B  10      -0.169  -3.939   4.305  1.00  0.00           H  
ATOM    138  HA3 GLY B  10      -1.387  -2.946   5.106  1.00  0.00           H  
TER     139      GLY B  10                                                      
ENDMDL                                                                          
MODEL        2                                                                  
ATOM      1  N   GLY A   1      -7.505  -1.771   3.409  1.00  0.00           N  
ATOM      2  CA  GLY A   1      -7.108  -0.535   2.677  1.00  0.00           C  
ATOM      3  C   GLY A   1      -5.810  -0.788   1.910  1.00  0.00           C  
ATOM      4  O   GLY A   1      -5.080  -1.717   2.192  1.00  0.00           O  
ATOM      5  H1  GLY A   1      -6.981  -2.584   3.026  1.00  0.00           H  
ATOM      6  H2  GLY A   1      -8.526  -1.930   3.292  1.00  0.00           H  
ATOM      7  H3  GLY A   1      -7.282  -1.664   4.419  1.00  0.00           H  
ATOM      8  HA2 GLY A   1      -7.891  -0.262   1.984  1.00  0.00           H  
ATOM      9  HA3 GLY A   1      -6.954   0.268   3.382  1.00  0.00           H  
ATOM     10  N   TYR A   2      -5.516   0.034   0.939  1.00  0.00           N  
ATOM     11  CA  TYR A   2      -4.264  -0.159   0.153  1.00  0.00           C  
ATOM     12  C   TYR A   2      -3.152   0.704   0.747  1.00  0.00           C  
ATOM     13  O   TYR A   2      -3.176   1.914   0.651  1.00  0.00           O  
ATOM     14  CB  TYR A   2      -4.507   0.257  -1.299  1.00  0.00           C  
ATOM     15  CG  TYR A   2      -3.235   0.093  -2.093  1.00  0.00           C  
ATOM     16  CD1 TYR A   2      -2.861  -1.171  -2.566  1.00  0.00           C  
ATOM     17  CD2 TYR A   2      -2.431   1.207  -2.362  1.00  0.00           C  
ATOM     18  CE1 TYR A   2      -1.684  -1.320  -3.308  1.00  0.00           C  
ATOM     19  CE2 TYR A   2      -1.252   1.058  -3.102  1.00  0.00           C  
ATOM     20  CZ  TYR A   2      -0.879  -0.206  -3.576  1.00  0.00           C  
ATOM     21  OH  TYR A   2       0.282  -0.353  -4.308  1.00  0.00           O  
ATOM     22  H   TYR A   2      -6.118   0.776   0.728  1.00  0.00           H  
ATOM     23  HA  TYR A   2      -3.973  -1.199   0.187  1.00  0.00           H  
ATOM     24  HB2 TYR A   2      -5.278  -0.364  -1.727  1.00  0.00           H  
ATOM     25  HB3 TYR A   2      -4.819   1.290  -1.330  1.00  0.00           H  
ATOM     26  HD1 TYR A   2      -3.481  -2.033  -2.358  1.00  0.00           H  
ATOM     27  HD2 TYR A   2      -2.719   2.182  -1.996  1.00  0.00           H  
ATOM     28  HE1 TYR A   2      -1.398  -2.295  -3.673  1.00  0.00           H  
ATOM     29  HE2 TYR A   2      -0.631   1.918  -3.307  1.00  0.00           H  
ATOM     30  HH  TYR A   2       0.171   0.120  -5.136  1.00  0.00           H  
ATOM     31  N   ASP A   3      -2.176   0.092   1.359  1.00  0.00           N  
ATOM     32  CA  ASP A   3      -1.061   0.880   1.956  1.00  0.00           C  
ATOM     33  C   ASP A   3      -0.103   1.322   0.841  1.00  0.00           C  
ATOM     34  O   ASP A   3       0.536   0.497   0.222  1.00  0.00           O  
ATOM     35  CB  ASP A   3      -0.304   0.005   2.958  1.00  0.00           C  
ATOM     36  CG  ASP A   3       0.697   0.865   3.732  1.00  0.00           C  
ATOM     37  OD1 ASP A   3       1.738   1.171   3.176  1.00  0.00           O  
ATOM     38  OD2 ASP A   3       0.404   1.204   4.867  1.00  0.00           O  
ATOM     39  H   ASP A   3      -2.174  -0.885   1.423  1.00  0.00           H  
ATOM     40  HA  ASP A   3      -1.462   1.742   2.464  1.00  0.00           H  
ATOM     41  HB2 ASP A   3      -1.007  -0.440   3.648  1.00  0.00           H  
ATOM     42  HB3 ASP A   3       0.225  -0.773   2.430  1.00  0.00           H  
ATOM     43  N   PRO A   4      -0.034   2.611   0.613  1.00  0.00           N  
ATOM     44  CA  PRO A   4       0.837   3.180  -0.429  1.00  0.00           C  
ATOM     45  C   PRO A   4       2.288   3.233   0.061  1.00  0.00           C  
ATOM     46  O   PRO A   4       3.218   3.179  -0.719  1.00  0.00           O  
ATOM     47  CB  PRO A   4       0.277   4.589  -0.635  1.00  0.00           C  
ATOM     48  CG  PRO A   4      -0.488   4.948   0.661  1.00  0.00           C  
ATOM     49  CD  PRO A   4      -0.809   3.618   1.367  1.00  0.00           C  
ATOM     50  HA  PRO A   4       0.761   2.614  -1.342  1.00  0.00           H  
ATOM     51  HB2 PRO A   4       1.085   5.287  -0.802  1.00  0.00           H  
ATOM     52  HB3 PRO A   4      -0.403   4.599  -1.472  1.00  0.00           H  
ATOM     53  HG2 PRO A   4       0.132   5.569   1.295  1.00  0.00           H  
ATOM     54  HG3 PRO A   4      -1.404   5.463   0.420  1.00  0.00           H  
ATOM     55  HD2 PRO A   4      -0.487   3.652   2.400  1.00  0.00           H  
ATOM     56  HD3 PRO A   4      -1.862   3.401   1.305  1.00  0.00           H  
ATOM     57  N   GLU A   5       2.486   3.337   1.347  1.00  0.00           N  
ATOM     58  CA  GLU A   5       3.873   3.391   1.886  1.00  0.00           C  
ATOM     59  C   GLU A   5       4.682   2.220   1.325  1.00  0.00           C  
ATOM     60  O   GLU A   5       5.819   2.372   0.924  1.00  0.00           O  
ATOM     61  CB  GLU A   5       3.832   3.295   3.414  1.00  0.00           C  
ATOM     62  CG  GLU A   5       3.979   4.691   4.023  1.00  0.00           C  
ATOM     63  CD  GLU A   5       2.599   5.228   4.410  1.00  0.00           C  
ATOM     64  OE1 GLU A   5       1.891   5.677   3.525  1.00  0.00           O  
ATOM     65  OE2 GLU A   5       2.275   5.179   5.585  1.00  0.00           O  
ATOM     66  H   GLU A   5       1.721   3.377   1.959  1.00  0.00           H  
ATOM     67  HA  GLU A   5       4.337   4.321   1.595  1.00  0.00           H  
ATOM     68  HB2 GLU A   5       2.889   2.865   3.721  1.00  0.00           H  
ATOM     69  HB3 GLU A   5       4.642   2.669   3.756  1.00  0.00           H  
ATOM     70  HG2 GLU A   5       4.603   4.635   4.904  1.00  0.00           H  
ATOM     71  HG3 GLU A   5       4.432   5.354   3.303  1.00  0.00           H  
ATOM     72  N   THR A   6       4.103   1.050   1.291  1.00  0.00           N  
ATOM     73  CA  THR A   6       4.838  -0.131   0.755  1.00  0.00           C  
ATOM     74  C   THR A   6       4.405  -0.389  -0.689  1.00  0.00           C  
ATOM     75  O   THR A   6       5.089  -1.050  -1.444  1.00  0.00           O  
ATOM     76  CB  THR A   6       4.520  -1.361   1.609  1.00  0.00           C  
ATOM     77  OG1 THR A   6       3.152  -1.707   1.446  1.00  0.00           O  
ATOM     78  CG2 THR A   6       4.799  -1.050   3.079  1.00  0.00           C  
ATOM     79  H   THR A   6       3.185   0.948   1.619  1.00  0.00           H  
ATOM     80  HA  THR A   6       5.899   0.063   0.784  1.00  0.00           H  
ATOM     81  HB  THR A   6       5.140  -2.186   1.295  1.00  0.00           H  
ATOM     82  HG1 THR A   6       2.874  -2.192   2.226  1.00  0.00           H  
ATOM     83 HG21 THR A   6       4.589  -1.923   3.677  1.00  0.00           H  
ATOM     84 HG22 THR A   6       4.169  -0.233   3.399  1.00  0.00           H  
ATOM     85 HG23 THR A   6       5.836  -0.772   3.197  1.00  0.00           H  
ATOM     86  N   GLY A   7       3.272   0.129  -1.080  1.00  0.00           N  
ATOM     87  CA  GLY A   7       2.797  -0.088  -2.475  1.00  0.00           C  
ATOM     88  C   GLY A   7       1.923  -1.342  -2.529  1.00  0.00           C  
ATOM     89  O   GLY A   7       1.783  -1.969  -3.561  1.00  0.00           O  
ATOM     90  H   GLY A   7       2.735   0.658  -0.456  1.00  0.00           H  
ATOM     91  HA2 GLY A   7       2.222   0.769  -2.796  1.00  0.00           H  
ATOM     92  HA3 GLY A   7       3.647  -0.217  -3.129  1.00  0.00           H  
ATOM     93  N   THR A   8       1.335  -1.714  -1.426  1.00  0.00           N  
ATOM     94  CA  THR A   8       0.470  -2.927  -1.416  1.00  0.00           C  
ATOM     95  C   THR A   8      -0.464  -2.882  -0.206  1.00  0.00           C  
ATOM     96  O   THR A   8      -0.343  -2.033   0.654  1.00  0.00           O  
ATOM     97  CB  THR A   8       1.347  -4.179  -1.338  1.00  0.00           C  
ATOM     98  OG1 THR A   8       2.682  -3.802  -1.028  1.00  0.00           O  
ATOM     99  CG2 THR A   8       1.320  -4.908  -2.682  1.00  0.00           C  
ATOM    100  H   THR A   8       1.461  -1.196  -0.604  1.00  0.00           H  
ATOM    101  HA  THR A   8      -0.117  -2.957  -2.320  1.00  0.00           H  
ATOM    102  HB  THR A   8       0.970  -4.836  -0.569  1.00  0.00           H  
ATOM    103  HG1 THR A   8       3.157  -4.591  -0.757  1.00  0.00           H  
ATOM    104 HG21 THR A   8       1.800  -4.296  -3.432  1.00  0.00           H  
ATOM    105 HG22 THR A   8       0.295  -5.091  -2.970  1.00  0.00           H  
ATOM    106 HG23 THR A   8       1.844  -5.848  -2.593  1.00  0.00           H  
ATOM    107  N   TRP A   9      -1.395  -3.793  -0.135  1.00  0.00           N  
ATOM    108  CA  TRP A   9      -2.338  -3.815   1.014  1.00  0.00           C  
ATOM    109  C   TRP A   9      -1.582  -4.213   2.284  1.00  0.00           C  
ATOM    110  O   TRP A   9      -1.201  -5.354   2.458  1.00  0.00           O  
ATOM    111  CB  TRP A   9      -3.438  -4.843   0.745  1.00  0.00           C  
ATOM    112  CG  TRP A   9      -3.807  -4.837  -0.706  1.00  0.00           C  
ATOM    113  CD1 TRP A   9      -3.282  -5.654  -1.650  1.00  0.00           C  
ATOM    114  CD2 TRP A   9      -4.778  -4.000  -1.388  1.00  0.00           C  
ATOM    115  NE1 TRP A   9      -3.871  -5.366  -2.867  1.00  0.00           N  
ATOM    116  CE2 TRP A   9      -4.803  -4.354  -2.757  1.00  0.00           C  
ATOM    117  CE3 TRP A   9      -5.632  -2.978  -0.953  1.00  0.00           C  
ATOM    118  CZ2 TRP A   9      -5.649  -3.713  -3.663  1.00  0.00           C  
ATOM    119  CZ3 TRP A   9      -6.486  -2.328  -1.859  1.00  0.00           C  
ATOM    120  CH2 TRP A   9      -6.496  -2.697  -3.212  1.00  0.00           C  
ATOM    121  H   TRP A   9      -1.475  -4.464  -0.840  1.00  0.00           H  
ATOM    122  HA  TRP A   9      -2.778  -2.837   1.142  1.00  0.00           H  
ATOM    123  HB2 TRP A   9      -3.082  -5.825   1.018  1.00  0.00           H  
ATOM    124  HB3 TRP A   9      -4.309  -4.601   1.337  1.00  0.00           H  
ATOM    125  HD1 TRP A   9      -2.525  -6.403  -1.480  1.00  0.00           H  
ATOM    126  HE1 TRP A   9      -3.668  -5.814  -3.714  1.00  0.00           H  
ATOM    127  HE3 TRP A   9      -5.625  -2.688   0.085  1.00  0.00           H  
ATOM    128  HZ2 TRP A   9      -5.651  -4.001  -4.704  1.00  0.00           H  
ATOM    129  HZ3 TRP A   9      -7.141  -1.543  -1.512  1.00  0.00           H  
ATOM    130  HH2 TRP A   9      -7.154  -2.195  -3.904  1.00  0.00           H  
ATOM    131  N   GLY B  10      -1.362  -3.283   3.173  1.00  0.00           N  
ATOM    132  CA  GLY B  10      -0.632  -3.613   4.430  1.00  0.00           C  
ATOM    133  C   GLY B  10      -1.635  -4.010   5.515  1.00  0.00           C  
ATOM    134  O   GLY B  10      -1.235  -4.090   6.664  1.00  0.00           O  
ATOM    135  OXT GLY B  10      -2.787  -4.225   5.177  1.00  0.00           O  
ATOM    136  H   GLY B  10      -1.678  -2.369   3.015  1.00  0.00           H  
ATOM    137  HA2 GLY B  10       0.046  -4.434   4.247  1.00  0.00           H  
ATOM    138  HA3 GLY B  10      -0.073  -2.750   4.760  1.00  0.00           H  
TER     139      GLY B  10                                                      
ENDMDL                      
MASTER      102    0    0    0    2    0    0    6   77    1    0    1          
END                                                                                                                                                                                  
"""

@pytest.fixture
def pdb_file():
    """Create a temporary PDB file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as f:
        f.write(PDB_STRING)
        filepath = f.name
    yield filepath
    pathlib.Path(filepath).unlink()


@pytest.fixture
def cif_file():
    """Create a temporary CIF file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cif", delete=False) as f:
        # A minimal CIF file content with required columns for Biotite
        f.write(
            """
data_test
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.pdbx_PDB_model_num
_atom_site.pdbx_PDB_ins_code
ATOM 1 N N GLY A 1 -6.778 -1.424 4.200 1.00 0.00 1 ?
"""
        )
        filepath = f.name
    yield filepath
    pathlib.Path(filepath).unlink()


@pytest.fixture
def hdf5_file(pdb_file):
    """Create a temporary HDF5 file."""
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        filepath = tmp.name
    traj = md.load_pdb(pdb_file)
    traj.save_hdf5(filepath)
    yield filepath
    pathlib.Path(filepath).unlink()


def test_af_to_mpnn():
    """Test conversion from AlphaFold to ProteinMPNN alphabet."""
    af_sequence = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    mpnn_sequence = af_to_mpnn(af_sequence)
    assert mpnn_sequence.tolist() == [
        0,
        14,
        11,
        2,
        1,
        13,
        3,
        5,
        6,
        7,
        9,
        8,
        10,
        4,
        12,
        15,
        16,
        18,
        19,
        17,
        20,
    ]


def test_mpnn_to_af():
    """Test conversion from ProteinMPNN to AlphaFold alphabet."""
    mpnn_sequence = np.array(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    )
    af_sequence = mpnn_to_af(mpnn_sequence)
    print(af_sequence)
    assert af_sequence.tolist() == [
      0,
      4,
      3,
      6,
      13,
      7,
      8,
      9,
      11,
      10,
      12,
      2,
      14,
      5,
      1,
      15,
      16,
      19,
      17,
      18,
      20,
    ]


def test_extend_coordinate():
    """Test the extend_coordinate function."""
    atom_a = np.array([0, 0, 0])
    atom_b = np.array([1, 0, 0])
    atom_c = np.array([1, 1, 0])
    bond_length = 1.5
    bond_angle = np.pi / 2
    dihedral_angle = np.pi / 2
    atom_d = extend_coordinate(atom_a, atom_b, atom_c, bond_length, bond_angle, dihedral_angle)
    assert_trees_all_close(atom_d, np.array([1., 1., 1.5]), atol=1e-6)


def test_compute_cb_precise():
    """Test the compute_cb_precise function."""
    n_coord = np.array([0, 0, 0])
    ca_coord = np.array([1.46, 0, 0])
    c_coord = np.array([1.46 + 1.52 * np.cos(111 * np.pi / 180), 1.52 * np.sin(111 * np.pi / 180), 0])
    cb_coord = compute_cb_precise(n_coord, ca_coord, c_coord)
    assert cb_coord.shape == (3,)


def test_string_key_to_index():
    """Test the string_key_to_index function."""
    key_map = {"A": 0, "B": 1, "C": 2}
    keys = np.array(["A", "C", "D"])
    indices = string_key_to_index(keys, key_map, unk_index=3)
    assert np.array_equal(indices, np.array([0, 2, 3]))


def test_string_to_protein_sequence():
    """Test the string_to_protein_sequence function."""
    sequence = "ARND"
    protein_seq = string_to_protein_sequence(sequence)
    expected = af_to_mpnn(np.array([0, 1, 2, 3]))
    assert np.array_equal(protein_seq, expected)


def test_protein_sequence_to_string():
    """Test the protein_sequence_to_string function."""
    protein_seq = af_to_mpnn(np.array([0, 1, 2, 3]))
    sequence = protein_sequence_to_string(protein_seq)
    assert sequence == "ARND"


def test_residue_names_to_aatype():
    """Test the residue_names_to_aatype function."""
    residue_names = np.array(["ALA", "ARG", "ASN", "ASP"])
    aatype = residue_names_to_aatype(residue_names)
    expected = af_to_mpnn(np.array([0, 1, 2, 3]))
    assert np.array_equal(aatype, expected)


def test_atom_names_to_index():
    """Test the atom_names_to_index function."""
    atom_names = np.array(["N", "CA", "C", "O", "CB"])
    indices = atom_names_to_index(atom_names)
    assert np.array_equal(indices, np.array([0, 1, 2, 4, 3]))


def test_atom_array_dihedrals():
    """Test the atom_array_dihedrals function."""
    pdb_path = StringIO(PDB_STRING)
    stack = strucarray(
        [
            Atom(
                coord=[1, 1, 1],
                atom_name="CA",
                res_name="GLY",
                res_id=1,
            )
        ]
    )
    dihedrals = atom_array_dihedrals(stack)
    assert dihedrals is None


def test_check_if_file_empty(tmp_path):
    """Test the _check_if_file_empty utility."""
    empty_file = tmp_path / "empty.txt"
    empty_file.touch()
    assert _check_if_file_empty(str(empty_file))

    non_empty_file = tmp_path / "non_empty.txt"
    non_empty_file.write_text("hello")
    assert not _check_if_file_empty(str(non_empty_file))

    assert _check_if_file_empty("non_existent_file.txt")


class TestParseInput:
    """Tests for the main `parse_input` function."""

    def test_parse_pdb_string(self):
        """Test parsing a PDB file from a string."""
        protein_stream = parse_input(StringIO(PDB_STRING))
        protein_list = list(protein_stream)
        assert len(protein_list) == 2
        protein = protein_list[0]
        assert isinstance(protein, ProteinTuple)
        assert protein.aatype.shape == (10,)
        assert protein.atom_mask.shape == (10, 37)
        assert protein.coordinates.shape == (10, 37, 3)
        assert protein.residue_index.shape == (10,)
        assert protein.chain_index.shape == (10,)
        assert protein.dihedrals is None

    def test_parse_pdb_file(self, pdb_file):
        """Test parsing a PDB file from a file path."""
        protein_stream = parse_input(pdb_file)
        protein_list = list(protein_stream)
        assert len(protein_list) == 2
        assert isinstance(protein_list[0], ProteinTuple)

    def test_parse_cif_file(self, cif_file):
        """Test parsing a CIF file from a file path."""
        protein_stream = parse_input(cif_file)
        protein_list = list(protein_stream)
        assert len(protein_list) == 1
        assert isinstance(protein_list[0], ProteinTuple)
        assert protein_list[0].aatype.shape == (1,)

    def test_parse_with_chain_id(self, pdb_file):
        """Test parsing with a specific chain ID."""
        protein_stream = parse_input(pdb_file, chain_id="A")
        protein_list = list(protein_stream)
        assert len(protein_list) == 2
        assert np.all(protein_list[0].chain_index == 0)

    def test_parse_with_invalid_chain_id(self, pdb_file):
        """Test parsing with an invalid chain ID."""
        with pytest.raises(RuntimeError, match="Failed to parse structure from source: AtomArray is empty."):
            list(parse_input(pdb_file, chain_id="Z"))

    def test_parse_empty_file(self):
        """Test parsing an empty file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=True) as tmp:
            with pytest.raises(RuntimeError):
                list(parse_input(tmp.name))

    def test_parse_empty_pdb_string(self):
        """Test parsing an empty PDB string."""
        with pytest.raises(RuntimeError, match="Failed to parse structure from source: Unknown file format."):
            list(parse_input(""))

    def test_parse_invalid_file(self):
        """Test parsing an invalid file path."""
        with pytest.raises(RuntimeError):
            list(parse_input("non_existent_file.pdb"))

    def test_parse_unsupported_format(self):
        """Test parsing an unsupported file format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
            tmp.write("hello")
            filepath = tmp.name
        with pytest.raises(RuntimeError, match="Failed to parse structure from source: Unknown file format '.txt'"):
            list(parse_input(filepath))
        pathlib.Path(filepath).unlink()

    def test_parse_mdtraj_trajectory(self, pdb_file):
        """Test parsing an mdtraj.Trajectory object."""
        traj = md.load_pdb(pdb_file)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".h5", delete=False) as tmp:
            traj.save_hdf5(tmp.name)
            filepath = tmp.name
        
        protein_stream = parse_input(filepath)
        protein_list = list(protein_stream)
        assert len(protein_list) == 2
        assert isinstance(protein_list[0], ProteinTuple)
        pathlib.Path(filepath).unlink()

    def test_parse_atom_array_stack(self):
        """Test parsing a biotite.structure.AtomArrayStack."""
        stack = AtomArrayStack(1, 4)
        stack.atom_name = np.array(["N", "CA", "C", "O"])
        stack.res_name = np.array(["GLY", "GLY", "GLY", "GLY"])
        stack.res_id = np.array([1, 1, 1, 1])
        stack.chain_id = np.array(["A", "A", "A", "A"])
        stack.coord = np.random.rand(1, 4, 3)
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as tmp:
            from biotite.structure.io.pdb import PDBFile
            pdb_file = PDBFile()
            pdb_file.set_structure(stack)
            pdb_file.write(tmp)
            filepath = tmp.name

        protein_stream = parse_input(filepath)
        protein_list = list(protein_stream)
        assert len(protein_list) == 1
        assert isinstance(protein_list[0], ProteinTuple)
        pathlib.Path(filepath).unlink()

    def test_parse_atom_array(self):
        """Test parsing a biotite.structure.AtomArray."""
        arr = AtomArray(4)
        arr.atom_name = np.array(["N", "CA", "C", "O"])
        arr.res_name = np.array(["GLY", "GLY", "GLY", "GLY"])
        arr.res_id = np.array([1, 1, 1, 1])
        arr.chain_id = np.array(["A", "A", "A", "A"])
        arr.coord = np.random.rand(4, 3)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as tmp:
            from biotite.structure.io.pdb import PDBFile
            pdb_file = PDBFile()
            pdb_file.set_structure(arr)
            pdb_file.write(tmp)
            filepath = tmp.name

        protein_stream = parse_input(filepath)
        protein_list = list(protein_stream)
        assert len(protein_list) == 1
        assert isinstance(protein_list[0], ProteinTuple)
        pathlib.Path(filepath).unlink()

    def test_parse_with_dihedrals(self):
        """Test parsing with dihedral angle extraction."""
        protein_stream = parse_input(StringIO(PDB_STRING), extract_dihedrals=True)
        protein_list = list(protein_stream)
        assert len(protein_list) == 2
        protein = protein_list[0]
        assert protein.dihedrals is not None
        assert protein.dihedrals.shape == (8, 3) # 10 resiudes - 2, not sure why first and last residues lack dihedrals but its something with biotite

    def test_parse_hdf5(self, hdf5_file):
        """Test parsing an HDF5 file."""
        protein_stream = parse_input(hdf5_file)
        protein_list = list(protein_stream)
        assert len(protein_list) == 2
        protein = protein_list[0]
        assert isinstance(protein, ProteinTuple)
        assert protein.aatype.shape == (10,)
        assert protein.atom_mask.shape == (10, 37)
        assert protein.coordinates.shape == (10, 37, 3)



