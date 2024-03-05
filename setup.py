from setuptools import setup
import versioneer

setup(
    packages=["bounds"],
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
)
