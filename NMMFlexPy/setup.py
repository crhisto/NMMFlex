import re
from pathlib import Path

from setuptools import setup, find_namespace_packages

HERE = Path(__file__).parent


def _read_version() -> str:
    """Single source of truth: NMMFlex.factorization.__version__."""
    text = (HERE / 'src' / 'NMMFlex' / 'factorization.py').read_text()
    match = re.search(r"^__version__\s*=\s*['\"]([^'\"]+)['\"]", text, re.M)
    if not match:
        raise RuntimeError("Could not locate __version__ in factorization.py")
    return match.group(1)


with open(HERE / 'src' / 'README.md', 'r') as f:
    long_description = f.read()

setup(
    name='NMMFlex',
    version=_read_version(),
    description='A Python package for NMMFlex which is an implementation of '
                'the Non-negative Multiple Matrix Factorization (NMMF) '
                'algorithm proposed in Takeuchi et al, 2013 with some '
                'improvements and modifications.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Crhistian Cardona',
    url='https://github.com/crhisto/NMMFlex',
    author_email='crhisto@gmail.com',
    package_dir={'': 'src'},
    packages=find_namespace_packages(where='src'),
    install_requires=(HERE / 'requirements.txt').read_text().splitlines(),
    extras_require={
        'torch': ['torch>=2.0'],
        'docs': ['sphinx>=7.0.1'],
        'dev': [
            'pytest>=7.3.1',
            'pytest-cov>=4.1.0',
            'ruff>=0.5.0',
            'torch>=2.0',
            'sphinx>=7.0.1',
        ],
    },
    test_suite="tests",
    python_requires='>=3.9',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)
