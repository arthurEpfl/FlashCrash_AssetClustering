# bigData

First create an environnement :
conda create --name test_env python=3.9



Steps to get the resulting clusters from our analysis in the juypter notebook : 
- Go to drive :  https://drive.google.com/drive/folders/1GOjfOQzDX0sPl-huZJgGX2NFV5F1rGdw?dmr=1&ec=wgc-drive-globalnav-goto&hl=fr
- And put the data in the data_clean folder
- Go into the jupyter notebook : covariance_matrix.ipynb
- Run the differents cells and choose if you just want to run the demo with the boolean argument in the main() definition 

To run the code directly from the main_subset.py on the sample subset:  
python main_subset.py --sample

The whole data from the google drive must be saved in processed/final_yearly

Note: running the code on the whole data takes multiple hours (3+).

To run the code on the whole data :
python main_subset.py 
