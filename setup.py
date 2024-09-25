import setuptools



__version__ = "0.0.0"

REPO_NAME = "end-to-end-classification"
AUTHOR_USER_NAME = "keerthikapopuri"
SRC_REPO = "cnnClassifier"
AUTHOR_EMAIL = "keerthikapopuri@gmail.com"


setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small python package for CNN app",
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "cnnClassifier"},
    packages=setuptools.find_packages(where="cnnClassifier")
)