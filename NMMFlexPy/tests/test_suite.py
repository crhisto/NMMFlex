import unittest

from tests.test_NMMFlex_basics import test_NMMFlex_basics
from tests.test_NMMFlex_checks import test_NMMFlex_checks
from tests.test_NMMFlex_deconvolution import test_NMMFlex_deconvolution
from tests.test_NMMFlex_gridsearch import test_NMMFlex_gridsearch
from tests.test_NMMFlex_regularization import test_NMMFlex_regularization
from tests.test_NMMFlex_sparseness import test_NMMFlex_sparseness
from tests.test_NMMFlex_sparseness_core import test_NMMFlex_sparseness_core
from tests.test_NMMFlex_sparseness_deconvolution \
    import test_NMMFlex_sparseness_deconvolution


# Run the following from the PyCharm virtual environment
# cd NMMFlexPy
# coverage run --source=src -m pytest --rootdir=. tests/test_suite.py
# pytest --cov=src tests/test_suite.py --cov-report=html
# Fix the problem with debugging after coverage running in PyCharm
# find . -name __pycache__ -type d -exec rm -rf {} \;

def suite():
    suite_object = unittest.TestSuite()
    suite_object.addTest(test_NMMFlex_basics(''))
    suite_object.addTest(test_NMMFlex_checks())
    suite_object.addTest(test_NMMFlex_deconvolution(''))
    suite_object.addTest(test_NMMFlex_gridsearch(''))
    suite_object.addTest(test_NMMFlex_regularization(''))
    suite_object.addTest(test_NMMFlex_sparseness())
    suite_object.addTest(test_NMMFlex_sparseness_core())
    suite_object.addTest(test_NMMFlex_sparseness_deconvolution())
    return suite_object


class testSuite_NMMF(unittest.TestSuite):
    if __name__ == '__main__':
        runner = unittest.TextTestRunner()
        runner.run(suite())
