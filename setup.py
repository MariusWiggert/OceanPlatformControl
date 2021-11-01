import setuptools 
import os

with open("README.md", "r", encoding="utf-8") as fh: 
    long_description = fh.read() 

setuptools.setup(
    name = 'ocean_navigation_simulator',
    version = '0.0.1',
    author = 'Marius Wiggert',
    author_email = 'mariuswiggert@berkeleuy.edu',
    description = 'Codebase to simulate ', # ?
    long_description = long_description, 
    long_description_content_type = 'text/markdown',
    url = 'https://github.com/MariusWiggert/OceanPlatformControl/',
    project_urls = {}, 
    license = 'MIT',
    packages = ['ocean_navigation_simulator'],
    python_requires='>3.9.1',
    # Included core requirements: https://packaging.python.org/discussions/install-requires-vs-requirements/
    install_requires=['jupyterlab', 'numpy', 'tqdm', 'pandas', 'seaborn',
                      'plotly', 'matplotlib', 'casadi', 'ffmpeg', 'imageio', 'netCDF4', 'datetime'],
)
