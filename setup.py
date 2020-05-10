from setuptools import setup

setup(
    name='sr549',
    version='0.0.1',
    packages=['sr549'],
    # package_dir={'sr549': 'python'},
    author="Evan Widloski, Aditya Deshmukh, Akshayaa Magesh",
    author_email="evan@evanw.org",
    description="ECE549 Super-resolution project",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    license="GPLv3",
    keywords="super resolution",
    url="https://github.com/evidlo/sr549",
    install_requires=[
        'matplotlib',
        'numpy',
        'imageio',
        'scipy',
        'scikit-image',
        'tqdm',
        'cachalot'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
    ]
)
