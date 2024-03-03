
import importlib.util
# Function to check if skmultilearn is installed
def is_package_installed(package_name, / ):
    return importlib.util.find_spec(package_name) is not None