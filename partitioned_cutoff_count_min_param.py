import os
import sys
import time
import argparse
import numpy as np
from itertools import repeat

from multiprocessing import Pool
from utils.utils import get_stat, git_log, get_data_str_with_ports_list
from utils.aol_utils import get_data_aol_query_list
from sketch import cutoff_countmin, cutoff_lookup, cutoff_countmin_wscore, order_y_wkey_list
from sketch import cutoff_countsketch, cutoff_countsketch_wscore
from sketch import count_min_partitioned, count_sketch, count_min
from optimization import getSizeProportionsFromThresholds

rangeMax = 3129.52612
# rangeMin = -37.3439026 # Will have to round up negative values here

def cutoff_countmin_p(y, scores, score_start, score_end, n_cm_buckets, n_hashes):
    """ Learned Partitioned Count-Min (use predicted scores to identify heavy hitters)
        Args:
        y: true counts of each item (sorted, largest first), float - [num_items]
        scores: predicted scores of each item - [num_items]
        score_cutoff: threshold for heavy hitters
        n_cm_buckets: number of buckets of Count-Min
        n_hashes: number of hash functions

    Returns:
        loss: estimation error
        count: number of elements in this bucket
        space: space usage in bytes
    """
    if len(y) == 0:
        raise TypeError("The length of y is 0")
        # avoid division of 0
    assert len(scores) == len(y)


    y_mid = []
    for i in range(len(y)):
        if scores[i] >= score_start and scores[i] < score_end:
            y_mid.append(y[i])
    
    # print('y_higher: %s, y_lower: %s, y_mid: %s' % (score_start, score_end, str(len(y_mid))))
    loss_cm = count_min(y_mid, n_cm_buckets, n_hashes)
    loss = loss_cm * np.sum(y_mid)
    space = n_cm_buckets * n_hashes * 4
    return loss, space

def run_ccm_p(y, scores, score_start, score_end, n_cm_buckets, n_hashes, name):
    start_t = time.time()
    loss, space = cutoff_countmin_p(y, scores, score_start, score_end, n_cm_buckets, n_hashes)
    # f = open('filename.txt', 'a')
    print('thresholdstart:%s, thresholdend:%s, # hashes %s, # space: %s # cm buckets %s - loss %s time: %s sec ' % \
    (str(score_start), str(score_end), str(n_hashes), str(space), str(n_cm_buckets), str(loss), str(time.time() - start_t)))
    # f.close()
    return loss, space

# def run_ccmp_wscore(y, scores, score_cutoff, n_cm_buckets, n_hashes, name, sketch_type):
#     start_t = time.time()
#     loss, space = cutoff_countmin_wscore(y, scores, score_cutoff, n_cm_buckets, n_hashes)
#     print('%s: scut: %.3f, # hashes %d, # cm buckets %d - loss %.2f\t time: %.2f sec' % \
#         (name, score_cutoff, n_hashes, n_cm_buckets, loss, time.time() - start_t))
#     return loss, space



if __name__ == '__main__':
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--test_results", type=str, nargs='*', help="testing results of a model (.npz file)", default='')
    argparser.add_argument("--valid_results", type=str, nargs='*', help="validation results of a model (.npz file)", default='')
    argparser.add_argument("--test_data", type=str, nargs='*', help="list of input .npy data", required=True)
    argparser.add_argument("--valid_data", type=str, nargs='*', help="list of input .npy data", required=True)
    argparser.add_argument("--lookup_data", type=str, nargs='*', help="list of input .npy data", default=[])
    argparser.add_argument("--save", type=str, help="prefix to save the results", required=True)
    argparser.add_argument("--seed", type=int, help="random state for sklearn", default=69)
    argparser.add_argument("--space_list", type=float, nargs='*', help="space in MB", default=[])
    argparser.add_argument("--n_hashes_list", type=int, nargs='*', help="number of hashes", required=True)
    argparser.add_argument("--perfect_order", action='store_true', default=False)
    argparser.add_argument("--n_workers", type=int, help="number of workers", default=10)
    argparser.add_argument("--aol_data", action='store_true', default=False)
    argparser.add_argument("--count_sketch", action='store_true', default=False)
    argparser.add_argument("--k", type=int, required=True)
    argparser.add_argument("--bcutoff", type=float, nargs='*', required=True)
    args = argparser.parse_args()

    assert not (args.perfect_order and args.lookup_data),   "use either --perfect or --lookup"

    k = args.k
    b_cutoffs = args.bcutoff
    assert len(b_cutoffs) == k + 1

    command = ' '.join(sys.argv) + '\n'
    log_str = command
    log_str += git_log() + '\n'
    print(log_str)
    np.random.seed(args.seed)

    name = "partitioned_count_min_sketch"

    start_t = time.time()
    if args.aol_data:
        x_valid, y_valid = get_data_aol_query_list(args.valid_data)
        x_test, y_test = get_data_aol_query_list(args.test_data)
    else:
        x_valid, y_valid = get_data_str_with_ports_list(args.valid_data)
        x_test, y_test = get_data_str_with_ports_list(args.test_data)
    log_str += get_stat('valid data:\n'+'\n'.join(args.valid_data), x_valid, y_valid)
    log_str += get_stat('test data:\n'+'\n'.join(args.test_data), x_test, y_test)

    if args.lookup_data:
        if args.aol_data:
            x_train, y_train = get_data_aol_query_list(args.lookup_data)
        else:
            x_train, y_train = get_data_str_with_ports_list(args.lookup_data)
        log_str += get_stat('lookup data:\n'+'\n'.join(args.lookup_data), x_train, y_train)
    print('data loading time: %.1f sec' % (time.time() - start_t))

    if args.valid_results:
        key = 'valid_output'
        y_valid_ordered, valid_scores = order_y_wkey_list(y_valid, args.valid_results, key)

    if args.test_results:
        key = 'test_output'
        y_test_ordered, test_scores = order_y_wkey_list(y_test, args.test_results, key)

    if args.perfect_order:
        assert np.abs(1 - len(y_valid) / len(y_test)) < 0.1,   "valid and test data should have similar # items"
    
    countmin = []
    f = open('optimizedValues.txt', 'a')
    # print("bcut:" + " ".join([str(b) for b in b_cutoffs], file=f))
    print("kVal" + str(k) + " = ", end ="", file=f)
    print("[", file=f, end = "")
    
    # sizes = [1/20, 4/20, 15/20]
    # sizes = getSizeProportionsFromThresholds(b_cutoffs[1:3], 1.5, k)
    # print(sizes)
    # buckets = []
    # for exp in range(10): 
    #     proportionArr = [pow(i, exp) for i in range(1, k + 1)]
    #     sizes = [i/sum(proportionArr) for i in proportionArr] # normalize
    for i, val in enumerate(valid_scores):
        if val < 0: valid_scores[i] = 0
    for space in args.space_list:
        prev_bcut =  b_cutoffs[0:len(b_cutoffs)-1]
        next_bcut = b_cutoffs[1:]
        for n_hash in args.n_hashes_list:
            # buckets = []
            # for i in sizes:
            n_cmin_buckets = int(space * 1e6/ (n_hash * 4 * k)) 
            #     buckets.append()
            # Naive space per sketch: every bucket gets equal space 
            # n_cmin_buckets = int((space * 1e6 - bcut * 4 * cutoff_cost_mul) / (n_hash * 4))
            # nh_all.append(n_hash)
            # nb_all.append(n_cmin_buckets)
            # Calculate the loss of the one elements: 
            inY = []
            inScore = []
            notInY = []
            notInScore = []
            for i in range(len(y_valid_ordered)):
                if valid_scores[i] <= .50:
                    notInY.append(y_valid_ordered[i])
                    notInScore.append(valid_scores[i]) 
                else:
                    inY.append(y_valid_ordered[i])
                    inScore.append(valid_scores[i])

            loss = 0
            for i in range(len(notInY)):
                loss += np.abs(notInY[i] - 1) * notInY[i]
            loss / np.sum(notInY)

            start_t = time.time()
            pool = Pool(args.n_workers)
            results = pool.starmap(run_ccm_p, zip(repeat(inY), repeat(inScore), prev_bcut, next_bcut, repeat(n_cmin_buckets), repeat(n_hash), repeat(name)))
            pool.close()
            pool.join()
            loss_results, space_actual = zip(*results)
            bigE = np.sum(loss_results) / np.sum(y_valid_ordered)
            # print(np.sum(y_valid_ordered))
            # print(len(y_valid_ordered))
            char = "]" if space == args.space_list[-1] and n_hash == args.n_hashes_list[-1] else ","
            print(str(bigE) + char, file=f)
            # loss_results, sumResults, space_actual = zip(*results)
            # # weighted sum of error 
            # print(np.sum(sumResults))
            # # print("Length" + str()
            # print(np.sum(space_actual))
            # print("Space: " + str(space))
            # char = "]" if space == args.space_list[-1] and n_hash == args.n_hashes_list[-1] else ","
            # print("Big E:" + str(bigE) + char, file=f)
            # print("SpaceSum: " + str(np.sum(space_actual)), file=f)
            # print("====================================", file=f)
            # print("Space: " + str(space) + " n_hash: " + str(n_hash), file=f)
            # print("====================================", file=f)
            # print("Big error: " + str(bigE), file=f)
            # loss_countmin = count_min(y_valid_ordered, int(space * 1e6 / (n_hash * 4)) , n_hash)
            # # countmin.append(loss_countmin)
            # print("Count-min sketch error rate: " + str(loss_countmin),file=f)
            # print("====================================", file=f)
            # print("====================================", file=f)
        # print("Space: " + str(sizes), file=f)
    print("\n\n", file=f)
    print("\n\n", file=f)