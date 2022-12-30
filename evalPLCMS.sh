tday=50

# for (i=11; i<17l i++)
# do
#     for (j=i; j < 21; j++)
#     do
python3 partitioned_cutoff_count_min_param.py \
    --count_sketch \
    --space_list 0.1 .2 .3 .4 0.5 .6 .7 .8 .9 1.0 1.2\
    --n_hashes 1\
    --save csketch_aol_tday${tday}_u256 --n_workers 30 \
    --test_data     ./data/aol/query_counts/query_counts_day_00${tday}.npz \
    --valid_data    ./data/aol/query_counts/query_counts_day_0005.npz \
    --test_result   paper_predictions/aol_inf_all_v05_t${tday}_exp22_aol_5d_r1-h1_u256-32_eb64_bs128_ra_20180514-160509_ep190_res.npz \
    --valid_result  paper_predictions/aol_inf_all_v05_t06_exp22_aol_5d_r1-h1_u256-32_eb64_bs128_ra_20180514-160509_ep190_res.npz \
    --k 3\
    --bcutoff 0 1.52325165 1.73732948 3129.52612 \
    --aol
#     done
# done
# done
# python3 partitioned_cutoff_count_min_param.py.py \
#     --count_sketch \
#     --space_list 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 1.2 1.4 1.6 1.8 2 \
#     --n_hashes 4\
#     --save csketch_aol_tday${tday}_u256 --n_workers 30 \
#     --test_data     ./data/aol/query_counts/query_counts_day_00${tday}.npz \
#     --valid_data    ./data/aol/query_counts/query_counts_day_0005.npz \
#     --test_result   paper_predictions/aol_inf_all_v05_t${tday}_exp22_aol_5d_r1-h1_u256-32_eb64_bs128_ra_20180514-160509_ep190_res.npz \
#     --valid_result  paper_predictions/aol_inf_all_v05_t06_exp22_aol_5d_r1-h1_u256-32_eb64_bs128_ra_20180514-160509_ep190_res.npz \
#     --k 4\
#     --bcutoff 0 1.44332719 1.58760667 1.80719066 3129.52612 \
#     --aol

# python3 partitioned_cutoff_count_min_param.py.py \
#     --count_sketch \
#     --space_list 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 1.2 1.4 1.6 1.8 2 \
#     --n_hashes 4\
#     --save csketch_aol_tday${tday}_u256 --n_workers 30 \
#     --test_data     ./data/aol/query_counts/query_counts_day_00${tday}.npz \
#     --valid_data    ./data/aol/query_counts/query_counts_day_0005.npz \
#     --test_result   paper_predictions/aol_inf_all_v05_t${tday}_exp22_aol_5d_r1-h1_u256-32_eb64_bs128_ra_20180514-160509_ep190_res.npz \
#     --valid_result  paper_predictions/aol_inf_all_v05_t06_exp22_aol_5d_r1-h1_u256-32_eb64_bs128_ra_20180514-160509_ep190_res.npz \
#     --k 6\
#     --bcutoff 0 1.37867045 1.49402583 1.58760667 1.71090412 1.96081018 3129.52612 \
#     --aol

# python3 partitioned_cutoff_count_min_param.py.py \
#     --count_sketch \
#     --space_list 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 1.2 1.4 1.6 1.8 2 \
#     --n_hashes 4\
#     --save csketch_aol_tday${tday}_u256 --n_workers 30 \
#     --test_data     ./data/aol/query_counts/query_counts_day_00${tday}.npz \
#     --valid_data    ./data/aol/query_counts/query_counts_day_0005.npz \
#     --test_result   paper_predictions/aol_inf_all_v05_t${tday}_exp22_aol_5d_r1-h1_u256-32_eb64_bs128_ra_20180514-160509_ep190_res.npz \
#     --valid_result  paper_predictions/aol_inf_all_v05_t06_exp22_aol_5d_r1-h1_u256-32_eb64_bs128_ra_20180514-160509_ep190_res.npz \
#     --k 8\
#     --bcutoff 0 1.33332491 1.44332719 1.51735091 1.58760667 1.67447352 1.80719066 2.09107709 3129.52612 \
#     --aol

# python3 partitioned_cutoff_count_min_param.py.py \
#     --count_sketch \
#     --space_list 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 1.2 1.4 1.6 1.8 2 \
#     --n_hashes 4\
#     --save csketch_aol_tday${tday}_u256 --n_workers 30 \
#     --test_data     ./data/aol/query_counts/query_counts_day_00${tday}.npz \
#     --valid_data    ./data/aol/query_counts/query_counts_day_0005.npz \
#     --test_result   paper_predictions/aol_inf_all_v05_t${tday}_exp22_aol_5d_r1-h1_u256-32_eb64_bs128_ra_20180514-160509_ep190_res.npz \
#     --valid_result  paper_predictions/aol_inf_all_v05_t06_exp22_aol_5d_r1-h1_u256-32_eb64_bs128_ra_20180514-160509_ep190_res.npz \
#     --k 10\
#     --bcutoff 0 1.29545355 1.4070003 1.47468281 1.53111172 1.58760667 1.65447783 1.74480319 1.88838863 2.20702863 3129.52612 \
#     --aol

# python3 partitioned_cutoff_count_min_param.py.py \
#     --count_sketch \
#     --space_list 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 1.2 1.4 1.6 1.8 2 \
#     --n_hashes 4\
#     --save csketch_aol_tday50_u256 --n_workers 30 \
#     --test_data     ./data/aol/query_counts/query_counts_day_0050.npz \
#     --valid_data    ./data/aol/query_counts/query_counts_day_0005.npz \
#     --test_result   paper_predictions/aol_inf_all_v05_t50_exp22_aol_5d_r1-h1_u256-32_eb64_bs128_ra_20180514-160509_ep190_res.npz \
#     --valid_result  paper_predictions/aol_inf_all_v05_t06_exp22_aol_5d_r1-h1_u256-32_eb64_bs128_ra_20180514-160509_ep190_res.npz \
#     --k 12\
#     --bcutoff 0 1.26040578 1.37867045 1.44332719 1.49402583 1.5404222 1.58760667 1.64243126 1.71090412 1.80719066 1.96081018 2.31816554 3129.52612 \
#     --aol
