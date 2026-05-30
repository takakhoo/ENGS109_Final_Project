"""Sparse, Low-Rank, and Deep: a compressed-sensing toolkit for music source separation.

ENGS 109 (Spring 2026) final project, Taka Khoo.

Five separation stages, each tied to a course lecture:
  Stage A  Robust PCA                 L13  stage_a_rpca
  Stage B  Class-conditional NMF      L15  stage_b_nmf
  Stage C  K-SVD + SRC                L14, L8  stage_c_ksvd
  Stage D  SparseNet mask predictor   L16  stage_d_scn
  Stage E  MMV / SOMP stereo          L12  stage_e_mmv
"""

__version__ = "0.1.0"
__author__ = "Taka Khoo"
