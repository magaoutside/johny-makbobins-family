"""Create isolated Python 3.11 environments; heavy GPU dependencies are opt-in."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', action='store_true', help='Install all image-extraction environments (Linux/CUDA)')
    a = p.parse_args()
    if sys.version_info[:2] != (3, 11):
        p.error('Run with Python 3.11: python3.11 setup_env.py')
    environments = [('.venv', 'requirements-detectors.txt' if a.gpu else 'requirements.txt')]
    if a.gpu:
        environments += [('.venv-vlm', 'requirements-wevis.txt'), ('.venv-sam3', 'requirements-sam3.txt')]
        if (ROOT/'requirements-sam2.txt').exists():
            environments.append(('.venv-sam2', 'requirements-sam2.txt'))
    for name, requirement in environments:
        folder = ROOT/name
        subprocess.run([sys.executable, '-m', 'venv', str(folder)], check=True)
        python = folder/('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
        subprocess.run([str(python), '-m', 'pip', 'install', '-r', str(ROOT/requirement)], check=True)
        subprocess.run([str(python), '-m', 'pip', 'check'], check=True)
    print('Ready. Activate .venv, then follow README.md.')


if __name__ == '__main__':
    main()
