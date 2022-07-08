from moffragmentor.fragmentor.branching_points import get_branch_points

def test_get_branch_points(abaxin): 
    """Test the branch point detection on a case that previously failed (#70)"""
    bp = get_branch_points(abaxin)
    assert set(bp) == set([146, 147, 72, 73, 112, 113, 52, 53, 92, 93])