import dynamiqs as dq
from dynamiqs._checks import _warn_non_normalised
from ..order import TEST_INSTANT
import pytest
import warnings

@pytest.mark.run(order=TEST_INSTANT)
class TestSESolveEuler():

    def test__warn_non_normalised(self):
        psi = dq.fock(4, 0) + dq.fock(4, 1)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_non_normalised(psi, "state")
            assert len(w)==1
            assert issubclass(w[-1].category,UserWarning)
            

    def test__warn_non_normalised2(self):
        psi = dq.fock_dm(4, 0) + dq.fock_dm(4, 1) + dq.fock_dm(4, 2)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_non_normalised(psi, "state")
            assert len(w)==1
            assert issubclass(w[-1].category,UserWarning)
        

    def test__warn_non_normalised3(self):
        psi = dq.fock(4, 2)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_non_normalised(psi, "state")
            assert len(w)==0   
    
        


