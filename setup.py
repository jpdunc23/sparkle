from setuptools import setup

setup(name='sparkle',
      version='0.1',
      description='Sparse Kernel Learning in PySpark',
      url='http://github.com/jpdunc23/sparkle',
      author='James Duncan',
      author_email='jpduncan@berkeley.edu',
      license='MIT',
      packages=['sparkle'],
      install_requires=['numpy>=1.15.4', 'scipy>=1.2.0'],
      zip_safe=False)
