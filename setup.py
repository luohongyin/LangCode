from setuptools import setup, find_packages

setup(
    name='langpy-notebook',
    version='0.4',
    packages=find_packages(),
    install_requires=[
        # Any dependencies your project needs, e.g.
        # 'requests',
        'IPython',
        'notebook==6.4.8'
    ],
    author='Hongyin Luo',
    author_email='hyluo@mit.edu',
    description='Chat in Python on IPython notebook.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/luohongyin/langpy-notebook',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
