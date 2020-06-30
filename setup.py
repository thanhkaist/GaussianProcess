from distutils.core import setup
setup(
  name = 'gp',        
  packages = ['gp'],   
  version = '0.1',      
  license='MIT',       
  description = 'Simple gaussian process',   
  author = 'Thanh Nguyen',               
  author_email = 'thanhnguyen.kaist.ac.kr',     
  url = 'https://github.com/thanhkaist/GaussianProcess',   
  download_url = 'https://github.com/thanhkaist/GaussianProcess/archive/v0.1',    
  keywords = ['gaussian', 'process'],   
  install_requires=[            
          'numpy',
          'matplotlib',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)
