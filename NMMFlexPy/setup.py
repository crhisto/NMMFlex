from setuptools import setup

with open('NMMFlex/README.md', 'r') as f:
    long_description = f.read()

setup(
    name='NMMFlex',
    version='0.1.0',
    description='A Python package for NMMFlex which is an implementation of '
                'the Non-negative Multiple Matrix Factorization (NMMF) '
                'algorithm proposed in Takeuchi et al, 2013 with some '
                'improvements and modifications.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Crhistian Cardona',
    url='https://github.com/crhisto/NMMFlex',
    author_email='crhisto@gmail.com',
    packages=['NMMFlex'],
    install_requires=open('requirements.txt').readlines(),
    test_suite="tests",
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: GNU License',
        'Programming Language :: Python :: 3',
        'Topic :: Software Development :: Libraries :: Python Modules',
        # Add any other relevant classifiers
    ],
)
