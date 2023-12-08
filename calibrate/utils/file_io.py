import os
import os.path as osp
import zipfile
from typing import List


def mkdir(output_dir : str) -> None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def zipdir(path, result_path):
    zipf = zipfile.ZipFile(result_path, "w")
    for root, dirs, files in os.walk(path):
        for file in files:
            zipf.write(
                osp.join(root, file),
                osp.relpath(osp.join(root, file), osp.join(path, '..'))
            )
    zipf.close()


def load_list(path: str) -> List[str]:
    """load list from text file"""
    assert osp.exists(path), "{} does not exist".format(path)

    ret = []
    with open(path, "r") as f:
        for line in f:
            ret.append(line.strip())
    return ret


def save_list(lines, path: str) -> None:
    with open(path, "w") as f:
        for line in lines:
            f.write("{}\n".format(line))
