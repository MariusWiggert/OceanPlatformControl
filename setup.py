import setuptools 
import os

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def _get_version():
    with open(os.path.join(_CURRENT_DIR, "ocean_navigation_simulator", "__init__.py")) as f:
        for line in f:
            if line.startswith("__version__") and "=" in line:
                version = line[line.find("=") + 1:].strip(" '\"\n")
                if version:
                    return version
        raise ValueError("`__version__` not defined in `ocean_navigation_simulator/__init__.py`")


with open("README.md", "r", encoding="utf-8") as fh: 
    long_description = fh.read() 

test_packages = ["pytest==7.1.3", "pytest-cov==3.0.0"]

setuptools.setup(
    name = 'ocean_navigation_simulator',
    version = _get_version(),
    author = 'Marius Wiggert',
    author_email = 'mariuswiggert@berkeleuy.edu',
    description = 'Codebase to simulate ocean platform navigation',
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    url = 'https://github.com/MariusWiggert/OceanPlatformControl/',
    project_urls = {}, 
    license = 'MIT',
    packages = setuptools.find_packages(),
    python_requires='>=3.9.7',
    extras_require={
        "test": test_packages,
    },
    # Included core requirements: https://packaging.python.org/discussions/install-requires-vs-requirements/

    # please install requirements manually from .txt
    #install_requires=['jupyterlab', 'numpy', 'tqdm', 'pandas', 'seaborn',
    #                  'plotly', 'matplotlib', 'casadi', 'ffmpeg', 'imageio', 'netCDF4', 'datetime'],
)
