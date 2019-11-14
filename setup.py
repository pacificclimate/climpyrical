import sys
from setuptools import setup
from setuptools.command.test import test as TestCommand



class PyTest(TestCommand):

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = ['-v', 'tests']
        self.test_suite = True

    def run_tests(self):
        # import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(self.test_args)
        sys.exit(errno)

setup(
    name="climpyrical",
    description="A spatial basis pattern reconstruction tool",
    keywords="geography fields regression climate meteorology",
    packages=['climpyrical'],
    version='0.1dev',
    url="http://www.pacificclimate.org/",
    author="Nic Annau",
    author_email="nannau@uvic.ca",
    zip_safe=True,
    scripts=[
        'climpyrical/datacube.py',
        'climpyrical/mask.py',
    ],
    install_requires=['numpy', 'shapely', 'geopandas', 'xarray'],
    tests_require=['pytest'],
    cmdclass={'test': PyTest},

    classifiers='''
Intended Audience :: Science/Research
License :: GNU General Public License v3 (GPLv3)
Operating System :: OS Independent
Programming Language :: Python :: 3.6
Topic :: Scientific/Engineering
Topic :: Software Development :: Libraries :: Python Modules'''.split('\n')
)
