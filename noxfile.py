import pathlib

import nox

REQUIREMENTS_TEST = [
    "pytest==7.*",
    "pytest-aiohttp==1.*",
]

THIS_DIR = str(pathlib.Path(__file__).parent)


def _install_test_requirements(session):
    session.run('pip', 'install', '-e', '.')
    session.run('pip', 'install', *REQUIREMENTS_TEST)


@nox.session(venv_backend="none")
def install_test_requirements(session):
    _install_test_requirements(session)


@nox.session(reuse_venv=True)
def test(session):
    _install_test_requirements(session)
    session.run('pytest', 'tests', '-rP', '-vv', *session.posargs, env={'RICH_TRACEBACK': '0', 'PYTHONPATH': THIS_DIR})
