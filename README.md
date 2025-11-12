Problem: papers >> time

Goal: Find most top-K interesting papers.

Method: Filtering to K<=100 -> K=20. 

Details: I use [`constants.py`](constants.py) to filter to authors whose work I find interesting (hopefully to roughly 100 papers per conference), then manually read abstracts to filter to top-20 most interesting.

### data

- https://huggingface.co/datasets/davidheineman/colm-2025
- https://huggingface.co/datasets/davidheineman/neurips-2025

### scraper

```sh
export OPENREVIEW_USER="..."
export OPENREVIEW_PASS="..."

pip install openreview-py

# example commands
python scraper.py --conference colm --years 2024 2025
python scraper.py --conference iclr --years 2026
python scraper.py --conference neurips --years 2026
```