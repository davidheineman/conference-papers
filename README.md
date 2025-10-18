Problem: papers >> time

Goal: Find most top-K interesting papers.

Method: Filtering to K<=100 -> K=20. 

Details: I use [`constants.py`](constants.py) to filter to authors whose work I find interesting (hopefully to roughly 100 papers per conference), then manually read abstracts to filter to top-20 most interesting.