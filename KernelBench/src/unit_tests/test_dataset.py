
import pytest
from src.dataset import get_code_hash

"""
Usage 
pytest test_dataset.py
"""


def test_get_code_hash():
    """
    Test collision and equivalence checking
    """

    code_snippet_batch_1_v1 = """
    import torch 
    # This is for a single batch
    '''
    Some random multi-line comment
    '''
    B = 1
    """
    
    code_snippet_batch_1_v2 = """
    import torch 
    '''
    More problem descriptions (updated)
    '''
    # low batch setting

    B = 1
    """

    code_snippet_batch_64 = """
    import torch 
    # This is for a single batch
    '''
    Some random multi-line comment
    '''
    B = 64
    """

    assert get_code_hash(code_snippet_batch_1_v1) == get_code_hash(code_snippet_batch_1_v2), \
        "Hash should be equal for semantically equivalent code with different comments"
    
    assert get_code_hash(code_snippet_batch_1_v1) != get_code_hash(code_snippet_batch_64), \
        "Hash should differ for code with different batch sizes"