import os
import errno

def is_directory(name):
    return os.path.isdir(name)


def is_file(name):
    return os.path.isfile(name)


def exists(name):
    return os.path.exists(name)


def list_all(root, filter_func=lambda x: True, full_path=False):
    """List all entities directly under 'dir_name' that satisfy 'filter_func'

    Args:
        root: Name of directory to start search.
        filter_func: function or lambda that takes path.
        full_path: If True will return results as full path including `root`.

    Returns:
        list of all files or directories that satisfy the criteria.

    """
    if not is_directory(root):
        raise Exception(f"Invalid parent directory '{root}'")
    matches = [x for x in os.listdir(root) if filter_func(os.path.join(root, x))]
    return [os.path.join(root, m) for m in matches] if full_path else matches


def list_subdirs(dir_name, full_path=False):
    """
    Equivalent to UNIX command:
      ``find $dir_name -depth 1 -type d``

    Args:
        dir_name: Name of directory to start search.
        full_path: If True will return results as full path including `root`.

    Returns:
        list of all directories directly under 'dir_name'.
    """
    return list_all(dir_name, os.path.isdir, full_path)


def list_files(dir_name, full_path=False):
    """
    Equivalent to UNIX command:
      ``find $dir_name -depth 1 -type f``

    Args:
        dir_name: Name of directory to start search.
        full_path: If True will return results as full path including `root`.

    Returns:
        list of all files directly under 'dir_name'.
    """
    return list_all(dir_name, os.path.isfile, full_path)


def find(root, name, full_path=False):
    """Search for a file in a root directory. Equivalent to:
      ``find $root -name "$name" -depth 1``

    Args:
        root: Name of root directory for find.
        name: Name of file or directory to find directly under root directory.
        full_path: If True will return results as full path including `root`.

    Returns:
        list of matching files or directories.
    """
    path_name = os.path.join(root, name)
    return list_all(root, lambda x: x == path_name, full_path)


def mkdir(root, name=None):
    """Make directory with name "root/name", or just "root" if name is None.

    Args:
        root: Name of parent directory.
        name: Optional name of leaf directory.

    Returns:
        Path to created directory.
    """
    target = os.path.join(root, name) if name is not None else root
    try:
        os.makedirs(target)
    except OSError as e:
        if e.errno != errno.EEXIST or not os.path.isdir(target):
            raise e
    return target
