from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize(['chess.pyx', 'mr.pyx'], annotate=True))
