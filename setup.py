from setuptools import setup

with open("README.org", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(name='ecalautoanalysis',
      version='1.0',
      description='Python package to analyse data from ECAL Automation workflows to the influxdb backend',
      long_description = long_description,
      url='https://github.com/XavierKervyn/CERN-CMS-ECAL',
      author='Simone Pigazzini',
      author_email='simone.pigazzini@cern.ch',
      license='GPLv3',
      packages=[
          'ecalautoanalysis'
      ],
      scripts=[], #TODO: add scripts
      classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Linux",
      ],
      python_requires=">=3.6",
      install_requires=[
      ]
)
