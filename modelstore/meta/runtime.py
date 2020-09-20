import getpass
import sys


def get_python_version():
    vers = sys.version_info
    return ".".join(str(x) for x in [vers.major, vers.minor, vers.micro])


def get_user():
    return getpass.getuser()
