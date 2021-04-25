from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(Path(__file__).stem)

import os
import datetime


def make_document():
    """生成模块脚本的说明文档。"""
    code_dir = Path(__file__).parent
    doc_path = code_dir.joinpath('document.txt')
    with doc_path.open('wt', encoding='utf8') as fout:
        fout.write(f'## {code_dir.name}\n\n')

    for py_file in sorted(code_dir.glob('**/*.py')):
        if py_file.stem == '__init__':
            py_name = py_file.parent.relative_to(code_dir).__str__().replace('\\', '/')
        else:
            py_name = py_file.relative_to(code_dir).__str__()[:-3].replace('\\', '/')

        doc_path.open('at', encoding='utf8').write(f'### {py_name}\n')
        module_name = py_name.replace('/', '.')
        os.system(f'python -m pydoc {module_name} >> {doc_path}')
        doc_path.open('at', encoding='utf8').write('\n')

    lines = doc_path.open('rt', encoding='utf8').readlines()
    with doc_path.open('wt', encoding='utf8') as fout:
        for line in lines:
            if line.startswith('[0m'):
                logger.info(repr(line))
                fout.write(line[4:])
                continue
            fout.write(line)

    logger.info('Make document.txt done.')


def make_help():
    """生成运行项目的帮助文档。"""
    code_dir = Path(__file__).parent
    doc_path = code_dir.joinpath('help.txt')
    with open(doc_path, 'wt', encoding='utf8') as fout:
        fout.write(f'## {code_dir.name}\n\n')

    for py_file in sorted(code_dir.glob('*.py')):
        py_name = py_file.relative_to(code_dir).__str__()[:-3].replace('\\', '/')
        doc_path.open('at', encoding='utf8').write(f'### {py_name}\n')
        os.system(f'python {py_name}.py --help >> {doc_path}')
        doc_path.open('at', encoding='utf8').write('\n\n')

    lines = doc_path.open('rt', encoding='utf8').readlines()
    with doc_path.open('wt', encoding='utf8') as fout:
        for line in lines:
            if line.startswith('[0m'):
                logger.info(repr(line))
                fout.write(line[4:])
                continue
            fout.write(line)

    logger.info('Make help.txt done.')


def make_requirements():
    """生成项目的依赖包。"""
    os.system('pipreqs . --encoding=utf8 --force')
    reqs = sorted(open('requirements.txt').readlines(), key=lambda x: x.lower())
    with open('requirements.txt', 'wt', encoding='utf8') as fout:
        for line in reqs:
            if line.startswith('~'):
                fout.write(f'# {line}')
            else:
                fout.write(line)
    logger.info('Make requirements.txt done.')


def make_gitignore():
    """生成git项目的忽略列表。"""
    with open('.gitignore', 'wt', encoding='utf8') as fout:
        for line in '.idea .git __pycache__ venv static log'.split():
            fout.write(f'{line}\n')
    logger.info('Make .gitignore done.')


def make_readme():
    """生成README文档。"""
    if Path('README.md').is_file():
        with open('README.md', 'at', encoding='utf8') as fout:
            version = datetime.datetime.now().strftime('%y.%m.%d')[1:].replace('.0', '.')
            fout.write(f'\n### v{version}\n')
    else:
        with open('README.md', 'wt', encoding='utf8') as fout:
            fout.write(f'## {Path(__file__).parent.name}\n\n')
            fout.write(f'## 版本\n')
            version = datetime.datetime.now().strftime('%y.%m.%d')[1:].replace('.0', '.')
            fout.write(f'\n### v{version}\n')
    logger.info('Make README.md done.')


def pip_install_requirements(reqspath=''):
    reqspath = reqspath or 'requirements.txt'
    for line in open(reqspath, encoding='utf8'):
        pkg = line.strip()
        os.system(f'pip install {pkg}')
        logger.info(f'pip install {pkg} done.')


if __name__ == "__main__":
    logger.info(__file__)
    import sys

    if len(sys.argv) == 2:
        pip_install_requirements(sys.argv[1])
    else:
        pip_install_requirements()

    os.system(
        r'pip install torch==1.7.0+cpu torchvision==0.8.0 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html')
    logger.info(f'pip install torch==1.7.0+cpu done.')
    # make_requirements()
    # make_gitignore()
    # make_readme()
    # make_help()
    # make_document()
