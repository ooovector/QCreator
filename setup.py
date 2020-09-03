from setuptools import setup, find_packages
import docs.version
def readme():
    with open('README.md') as f:
        return f.read()


def license():
    with open('LICENSE') as f:
        return f.read()

setup(name='qcreator',
      version=docs.version.__version__,
      author='Ivan Tsitsiln, Ilia Besedin',
      author_email='tsitsilinivan@gmail.com',
      maintainer='Ilia Besedin',
      maintainer_email='ilia.besedin@gmail.com',
      description='Python based framework for superconducting qubits designing '
                  'developed by members of the supercoducting metamaterials laboratory at '
                   'NUST MISIS',
      long_description=readme(),
      url='https://github.com/ooovector/QCreator',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Science/Research',
          'Programming Language :: Python :: 3 :: Only',
          'Programming Language :: Python :: 3.7',
          'Topic :: Scientific/Engineering'
      ],
      license=license(),
      # if we want to install without tests:
      packages=find_packages(exclude=["*.tests", "tests"]),
      #packages=find_packages(),
      install_requires=[
          'numpy>=1.10',
          'IPython>=4.0',
          'lmfit>=0.9.5',
          'scipy>=0.17',
      ]
      )
