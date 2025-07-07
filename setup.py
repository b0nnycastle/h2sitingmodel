"""
Setup script for H2 Station Siting Model
For backwards compatibility with older pip versions
"""

from setuptools import setup, find_packages

# Read requirements
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Read README
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='h2-station-siting',
    version='1.0.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='Optimization model for hydrogen refueling station placement',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/h2-station-siting',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.9',
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'h2-station-model=h2_station_siting.cli:main',
        ],
    },
)