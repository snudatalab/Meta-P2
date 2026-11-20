"""
Fast and Accurate Domain Adaptation for Irregular Tensor Decomposition (KDD 2024)
Fast and Accurate Domain Adaptation for Irregular and Regular Tensor Decomposition (TKDE 2026)

Authors:
- Junghun Kim (bandalg97@snu.ac.kr), Seoul National University
- Ka Hyun Park (kahyun.park@snu.ac.kr), Seoul National University
- Jun-Gi Jang (jungi@dgist.ac.kr), DGIST
- U Kang (ukang@snu.ac.kr), Seoul National University

This software may be used only for research evaluation purposes.

For other purposes (e.g., commercial), please contact the authors.

utils.py
    - utility functions
"""

import torch

def kron(A, B):
    """
    Compute kronecker product of A and B
    """
    return torch.einsum("ab,cd->acbd", A, B).view(A.size(0)*B.size(0),  A.size(1)*B.size(1))

def khatri(A, B):
    """
    Compute Khatri-Rao product of A and B
    """
    return torch.cat([kron(A[:, i].unsqueeze(1), B[:, i].unsqueeze(1)) for i in range(A.size(1))], 1)
