from setuptools import setup

setup(name='cosmicfish',
      version='0.2.0',
      description='A Fisher forecasting package for light relic cosmology.',
      keywords=['fisher forecast cosmology neutrino dark matter'], 
      url='https://github.com/ndeporzio/cosmicfish',
      author='Nicholas DePorzio, Julian Munoz, Cora Dvorkin, Linda Xu',
      author_email='nicholasdeporzio@g.harvard.edu',
      license='',
      packages=['cosmicfish'],
      include_package_data=True,
      zip_safe=False,
      test_suite='nose.collector',
      tests_require=['nose'],
      # dependency_links=['https://github.com/lesgourg/class_public'],
      scripts=[
        'bin/test-script',
        'bin/install_class.sh',  
      ], 
      install_requires=[
          'numpy',
          'matplotlib',
          'pandas',
          'seaborn', 
          'scipy'
      ])
