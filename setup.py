import os
import re
import setuptools
import versioneer

NAME = "statsplot"
AUTHOR = "Silas Kieser"
AUTHOR_EMAIL = "silas.kieser@gmail.com"
DESCRIPTION = "A package for most important plot for (biological)-sciences"
LICENSE = "MIT"
KEYWORDS = "plotting"
URL = "https://github.com/silask/" + NAME
README = ".github/README.md"
CLASSIFIERS = []
INSTALL_REQUIRES = []
ENTRY_POINTS = {}
SCRIPTS = []

HERE = os.path.dirname(__file__)


def read(file):
    with open(os.path.join(HERE, file), "r") as fh:
        return fh.read()



LONG_DESCRIPTION = read(README)

if __name__ == "__main__":
    setuptools.setup(
        name=NAME,
        version=versioneer.get_version(),
        cmdclass=versioneer.get_cmdclass(),
        packages=setuptools.find_packages(),
        author=AUTHOR,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type="text/markdown",
        license=LICENSE,
        keywords=KEYWORDS,
        url=URL,
        classifiers=CLASSIFIERS,
        install_requires=INSTALL_REQUIRES,
        entry_points=ENTRY_POINTS,
        scripts=SCRIPTS,
        include_package_data=True,
    )
