# bigData

Project in the context of FIN-525 Financial Big Data course.
Read the Financial_Big_Data_report.pdf for a comprehensive understanding of methods used and results obtained.

First create an environnement :
- conda create --name test_env python=3.9
- conda activate test_env
- pip install -r requirements.txt

Run this code for a mock run on a sample of 0.1% of the data :
- python main_subset.py --sample

Note: the results are insignificant on sampled data.

Steps to get the resulting clusters from our analysis in the juypter notebook : 
- Go to drive :  https://drive.google.com/drive/folders/1GOjfOQzDX0sPl-huZJgGX2NFV5F1rGdw?dmr=1&ec=wgc-drive-globalnav-goto&hl=fr
- The link is on google drive, since the data exceeds the 2 GB limit for other websites
- And put the data in the data_clean folder
- Go into the jupyter notebook : covariance_matrix.ipynb
- Run the differents cells and choose if you just want to run the demo with the boolean argument in the main() definition 

The whole data from the google drive must be saved in processed/final_yearly

To run the code on the whole data :
- python main_subset.py 

Note: running the code on the whole data takes multiple hours (3+).
