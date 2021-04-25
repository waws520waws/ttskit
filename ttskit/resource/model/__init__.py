# author: kuangdd
# date: 2021/4/25
"""
### model
__init__.py
ge2e.kuangdd.pt
mellotron.kuangdd-rtvc.pt
mellotron_hparams.json
waveglow.kuangdd.pt
"""
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(Path(__file__).stem)

if __name__ == "__main__":
    logger.info(__file__)
    for line in sorted(Path(__file__).parent.glob('*')):
        print(line.name)
