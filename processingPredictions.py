import numpy as np
import sys
data = np.load("./paper_predictions/aol_inf_all_v05_t06_exp22_aol_5d_r1-h1_u256-32_eb64_bs128_ra_20180514-160509_ep190_res.npz")
print(data.files)
row = data.files
np.set_printoptions(threshold=np.inf)
sys.stdout=open("predictionsDay06.txt","w")
for i in row:
    print("--------------------------")
    print(data[i])
sys.stdout.close()