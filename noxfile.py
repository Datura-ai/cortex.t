import pathlib

import nox

REQUIREMENTS_TEST = [
    "pytest==7.*",
]


def install_myself(session):
    """Install from source."""
    session.run('pip', 'install', '-e', '.')


THIS_DIR = str(pathlib.Path(__file__).parent)


@nox.session(reuse_venv=True)
def test(session):
    install_myself(session)
    session.run('pip', 'install', *REQUIREMENTS_TEST)
    session.run('pytest', 'tests', '-rP', '-vv', *session.posargs, env={'RICH_TRACEBACK': '0', 'PYTHONPATH': THIS_DIR})
