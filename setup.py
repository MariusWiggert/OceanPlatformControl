import setuptools 
import os

with open("README.md", "r", encoding="utf-8") as fh: 
    long_description = fh.read() 

setuptools.setup(
    name = 'ocean_platform_package', # add username
    version = '0.0.1',
    author = 'Example Author', # ? 
    author_email = 'author@example.com', # ? 
    description = 'Ocean Platform Package', # ? 
    long_description = long_description, 
    long_description_content_type = 'text/markdown',
    url = 'https://github.com/MariusWiggert/OceanPlatformControl/',
    project_urls = {}, 
    license = 'MIT',
    packages = ['ocean_platform_package'],
    install_requires=['jupyterlab', 'numpy', 'tqdm', 'pandas', 'seaborn', 'plotly', 'matplotlib', 'casadi', 'ffmpeg', 'imageio'],
)
