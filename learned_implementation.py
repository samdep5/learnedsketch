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
from sketch import cutoff_countsketch, cutoff_countsketch_wscore, count_min
from cutoff_count_min_param import get_great_cut, run_ccm
from partitioned_cutoff_count_min_param import run_ccm_p

rangeMax = 3129.52612
rangeMin = -37.3439026 # Will have to round up negative values here

def cutoff_countmin_p(y, scores, score_start, score_end, n_cm_buckets, n_hashes):
    if len(y) == 0:
        raise TypeError("The length of y is 0")
        # avoid division of 0
    assert len(scores) == len(y)
    if score_start == 0: score_start = rangeMin

    y_mid = []
    for i in range(len(y)):
        if scores[i] > score_start and scores[i] <= score_end:
            y_mid.append(y[i])

    
    # print('y_higher: %s, y_lower: %s, y_mid: %s' % (score_start, score_end, str(len(y_mid))))
    loss_cm = count_min(y_mid, n_cm_buckets, n_hashes)
    loss = loss_cm * (np.sum(y_mid) / np.sum(y))
    # print(np.sum(y))
    space = n_cm_buckets * n_hashes * 4
    return loss, space

def learned_partition(n_hash, k, y_valid_ordered, valid_scores, prev_bcut, next_bcut, n_cmin_buckets, y_total):
    f = open('filename.txt', 'a')
    # We assume that the space passed has already unincluded the space for the heavy hitters 
    proportionArr = None
    sizes = None
    for exp in range(10): 
        proportionArr = [pow(i, exp) for i in range(1, k + 1)]
        sizes = [i/sum(proportionArr) for i in proportionArr] # normalize
        buckets = [int(i * n_cmin_buckets) for i in sizes]
        # nh_all.append(n_hash)
        # nb_all.append(n_cmin_buckets)
        start_t = time.time()
        # pool = Pool(args.n_workers)
        loss_results = []
        space_actual = []
        for i in range(len(prev_bcut)):
            loss, space = run_ccm_p(y_valid_ordered, valid_scores, prev_bcut[i], next_bcut[i], buckets[i], n_hash, "learned_partition")
            loss_results.append(loss)
            space_actual.append(space)
    # results = pool.starmap(run_ccm_p, zip(repeat(y_valid_ordered), repeat(valid_scores), prev_bcut, next_bcut, repeat(n_cmin_buckets), repeat(n_hash)))
    # pool.close()
    # pool.join()
        f = open("output.txt", 'a')
        bigE = np.sum(loss_results) / np.sum(y_total)
        print("SPACE" + str((len(y_total) - len(y_valid_ordered)) * 4 * 2 + n_cmin_buckets * n_hash * 4), file = f)
        print("ERROR:" + str(bigE), file = f)
        print("SIZES: " + str(sizes), file = f)

    f.close
    # print(np.sum(y_valid_ordered))
    # print(len(y_valid_ordered))
    # bigE = bigE / np.sum(sumResults)
    # print("Big error: " + str(bigE), file=f)
    # print("====================================", file=f)
    # char = "]" if space == args.space_list[-1] and n_hash == args.n_hashes_list[-1] else ","
    return bigE
    # print("SpaceSum: " + str(np.sum(space_actual)), file=f)
    # print("====================================", file=f)
    # print("Space: " + str(space) + " n_hash: " + str(n_hash), file=f)
    # print("====================================", file=f)
    # print("Big error: " + str(bigE), file=f)
    # print("====================================", file=f)
    # val, num = count_min_partitioned(y_valid_ordered, int(space * 1e6 / (n_hash * 4)) , n_hash)
    # print("Count-min sketch error rate: " + str(val/num),file=f)
    # print("====================================", file=f)
    # f.close()


def cutoff_countmin_wscore_hybrid(y, scores, score_cutoff, n_cm_buckets, n_hashes):
    if len(y) == 0:
        return 0            # avoid division of 0
    y_ccm = y[scores >  score_cutoff]
    scoreSub = []
    y_cm = []
    for i in range(len(y)):
        if scores[i] <= score_cutoff:
            y_cm.append(y[i])
            scoreSub.append(scores[i])
    y_cmSort = sorted(y_cm)
    b_cutoff_starts = [0]
    b_cutoff_ends = []
    for i in range(1, k):
        b_cutoff_ends.append(y_cmSort[len(y_cm) * (i) // k])
        b_cutoff_starts.append(y_cmSort[i*len(y_cm) // k])
    b_cutoff_ends.append(y_cmSort[-1])
    
    loss_cf = 0  # put y_ccm into cutoff buckets, no loss
    loss_cm = learned_partition(n_hashes, k, y_cm, scoreSub, b_cutoff_starts, b_cutoff_ends, n_cm_buckets, y)

    assert len(y_ccm) + len(y_cm) == len(y)
    loss_avg =  np.sum(y_cm) / np.sum(y)
    # loss_avg = loss_avg / np.sum(y_cm)
    print('\tloss_cf %.2f\tloss_rd %.2f\tloss_avg %.2f' % (loss_cf, loss_cm, loss_avg))

    space = len(y_ccm) * 4 * 2 + n_cm_buckets * n_hashes * 4
    return loss_avg, space

def run_ccm_hybridp(y, scores, score_cutoff, n_cm_buckets, n_hashes):
    start_t = time.time()
        
    loss, space = cutoff_countmin_wscore_hybrid(y, scores, score_cutoff, n_cm_buckets, n_hashes)

    # f = open('filename.txt', 'a')
    print('# hashes %s, # space: %s # cm buckets %s - loss %s time: %s sec ' % \
    (str(n_hashes), str(space), str(n_cm_buckets), str(loss), str(time.time() - start_t)))
    # f.close()
    return loss, space



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
    args = argparser.parse_args()

    assert not (args.perfect_order and args.lookup_data),   "use either --perfect or --lookup"

    k = args.k

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


    cutoff_cost_mul = 2 # cutoff buckets cost x2
    bcut_all = []
    scut_all = []
    nh_all = []
    nb_all = []
    countmin = []
    f = open('filename.txt', 'a')
    # print("bcut:" + " ".join([str(b) for b in b_cutoffs], file=f))
    # print("kVal" + str(k) + " = ", end ="", file=f)
    # print("[", file=f, end = "")
    # sizes = getSizeProportionsFromThresholds(b_cutoffs[1:3], 1.5, k)
    # print(sizes)
    # buckets = []
    # for i, val in enumerate(valid_scores):
    #     if val < 0: valid_scores[i] = 0
    for space in args.space_list:
        max_bcut = space * 1e6 / (4 * cutoff_cost_mul)
        b_cutoffs = np.linspace(0.1, 0.9, 9) * max_bcut
        for bcut in b_cutoffs:
            for n_hash in args.n_hashes_list:
                bcut = int(bcut)
                if args.perfect_order:
                    scut = 0    # version 2, scut is not used
                elif args.lookup_data:
                    bcut, scut = get_great_cut(bcut, y_train, np.floor(max_bcut))    # this has to be y_train
                else:
                    if bcut < len(y_valid):
                        scut = valid_scores[bcut]
                    else:
                        scut = valid_scores[-1]
                n_cmin_buckets = int((space * 1e6 - bcut * 4 * cutoff_cost_mul) / (n_hash * 4))
                bcut_all.append(bcut)
                scut_all.append(scut)
                nh_all.append(n_hash)
                nb_all.append(bcut + n_cmin_buckets)
    rshape = (len(args.space_list), len(b_cutoffs), len(args.n_hashes_list))
    n_cm_all = np.array(nb_all) - np.array(bcut_all)
    
    # if args.lookup_data:
    #     min_scut = np.min(scut_all) # no need to store elements that are smaller
    #     x_train = np.asarray(x_train)
    #     x_train_hh = x_train[y_train > min_scut]
    #     y_train_hh = y_train[y_train > min_scut]
    #     lookup_dict = dict(zip(x_train_hh, y_train_hh))
    # sketch_type = "count-min"
    # start_t = time.time()
    # pool = Pool(args.n_workers)
    # results = pool.starmap(run_ccm, zip(repeat(y_valid_ordered), bcut_all, nh_all, nb_all, repeat(name), repeat(sketch_type)))
    # pool.close()
    # pool.join()

    best_space_list = np.array([.1,.2,.3,.4,.5,.6,.7,.8,.9,1,1.2,1.4,1.6,1.8,2])
    best_valid_loss = np.array([13.11,6.36,4.15,3.02,2.39,1.96,1.63,1.38,1.16,0.98,0.66,0.36,0.10,0.00,0.00])
    best_scuts = np.array([7.555,4.492,3.719,3.326,3.073,2.346,2.754,2.181,1.840,1.803,1.622,1.487,1.341,0.955,-44.703])
    best_bcuts = np.array([1250,2500,3750,5000,6250,15000,8750,20000,45000,50000,90000,140000,180000,202500,225000])
    best_n_buckets =np.array( [23750,47500,71250,95000,118750,135000,166250,180000,180000,200000,210000,210000,200000,213750,275000])
    best_n_hashes = np.array([1,1,1,1,1,1,1,1,1,1,1,1,2,4,1])
    best_valid_space = np.array([10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 120000, 140000, 160000, 180000, 200000])

    # valid_results, space_actual = zip(*results)
    # valid_results = np.reshape(valid_results, rshape)
    # space_actual = np.reshape(space_actual, rshape)
    # bcut_all = np.reshape(bcut_all, rshape)
    # scut_all = np.reshape(scut_all, rshape)
    # nh_all = np.reshape(nh_all, rshape)
    # nb_all = np.reshape(nb_all, rshape)
    # rshape = (len(args.space_list), -1)

    # best_param_idx = np.argmin(valid_results.reshape(rshape), axis=1)
    # best_scuts     = scut_all.reshape(rshape)[np.arange(rshape[0]), best_param_idx]
    # best_bcuts     = bcut_all.reshape(rshape)[np.arange(rshape[0]), best_param_idx]
    # best_n_buckets = nb_all.reshape(rshape)[np.arange(rshape[0]), best_param_idx]
    # best_n_hashes  = nh_all.reshape(rshape)[np.arange(rshape[0]), best_param_idx]
    # best_valid_loss  = valid_results.reshape(rshape)[np.arange(rshape[0]), best_param_idx]
    # best_valid_space = space_actual.reshape(rshape)[np.arange(rshape[0]), best_param_idx]

    # for i in range(len(best_valid_loss)):
    #     log_str += 'space: %.2f, scut %.3f, bcut %d, n_buckets %d, n_hashes %d - \tloss %.2f\tspace %.1f\n' % \
    #     (args.space_list[i], best_scuts[i], best_bcuts[i], best_n_buckets[i], best_n_hashes[i], best_valid_loss[i], best_valid_space[i])
 
#    t data using best parameters
    pool = Pool(args.n_workers)     
    results = pool.starmap(run_ccm_hybridp, zip(repeat(y_valid_ordered), repeat(valid_scores), best_scuts, best_n_buckets - best_bcuts, best_n_hashes))
    pool.close()
    pool.join()
    f = open("optimizedValues.txt", "a")
    test_results, space_test = zip(*results)
    print("test_results", file=f)
    print(test_results / np.sum(y_valid_ordered), file=f)
    print(space_test)
           
            # char = "]" if space == args.space_list[-1] and n_hash == args.n_hashes_list[-1] else ","
            # print(str(bigE) + char, file=f)
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
    # print("sup bitch")
    # print("Big error: " + str(bigE), file=f)
    # loss_countmin = count_min(y_valid_ordered, int(space * 1e6 / (n_hash * 4)) , n_hash, valid_scores)
    # countmin.append(loss_countmin)
    # print("Count-min sketch error rate: " + str(loss_countmin),file=f)
    # print("====================================", file=f)
            # print("====================================", file=f)
    print("\n\n", file=f)
    print("\n\n", file=f)


    # print("Learned CCM Hybrid with Partition beginning now")
    # cutoff_cost_mul = 2 # cutoff buckets cost x2
    # bcut_all = []
    # scut_all = []
    # nh_all = []
    # nb_all = []
    # for space in args.space_list:
    #     max_bcut = space * 1e6 / (4 * cutoff_cost_mul)
    #     b_cutoffs = np.linspace(0.1, 0.9, 9) * max_bcut
    #     for bcut in b_cutoffs:
    #         for n_hash in args.n_hashes_list:
    #             bcut = int(bcut)
    #             if args.perfect_order:
    #                 scut = 0    # version 2, scut is not used
    #             elif args.lookup_data:
    #                 bcut, scut = get_great_cut(bcut, y_train, np.floor(max_bcut))    # this has to be y_train
    #             else:
    #                 if bcut < len(y_valid):
    #                     scut = valid_scores[bcut]
    #                 else:
    #                     scut = valid_scores[-1]
    #             n_cmin_buckets = int((space * 1e6 - bcut * 4 * cutoff_cost_mul) / (n_hash * 4))
    #             bcut_all.append(bcut)
    #             scut_all.append(scut)
    #             nh_all.append(n_hash)
    #             nb_all.append(bcut + n_cmin_buckets)
    # rshape = (len(args.space_list), len(b_cutoffs), len(args.n_hashes_list))
    # n_cm_all = np.array(nb_all) - np.array(bcut_all)

    # pool = Pool(args.n_workers)
    # results = pool.starmap(run_ccm_wscore_hybrid,
    #         zip(repeat(y_test_ordered), repeat(test_scores), best_scuts, best_n_buckets - best_bcuts, best_n_hashes, repeat("countminwscorehybrid"), repeat("count-min")))

    # rshape = (len(args.space_list), -1)
    # best_param_idx = np.argmin(valid_results.reshape(rshape), axis=1)
    # best_scuts     = scut_all.reshape(rshape)[np.arange(rshape[0]), best_param_idx]
    # best_bcuts     = bcut_all.reshape(rshape)[np.arange(rshape[0]), best_param_idx]
    # best_n_buckets = nb_all.reshape(rshape)[np.arange(rshape[0]), best_param_idx]
    # best_n_hashes  = nh_all.reshape(rshape)[np.arange(rshape[0]), best_param_idx]
    # best_valid_loss  = valid_results.reshape(rshape)[np.arange(rshape[0]), best_param_idx]
    # best_valid_space = space_actual.reshape(rshape)[np.arange(rshape[0]), best_param_idx]
    # f.close()

    # rshape = (len(args.space_list), len(prev_bcut), len(len(args.n_hashes_list))
    # n_cm_all = np.array(nb_all) - np.array(bcut_all)
    # if args.lookup_data:
    #     min_scut = np.min(scut_all) # no need to store elements that are smaller
    #     x_train = np.asarray(x_train)
    #     x_train_hh = x_train[y_train > min_scut]
    #     y_train_hh = y_train[y_train > min_scut]
    #     lookup_dict = dict(zip(x_train_hh, y_train_hh))
    # print(nh_all)
    
    # if args.perfect_order:
    #     y_sorted = np.sort(y_valid)[::-1]
    #     results = pool.starmap(run_ccm, zip(repeat(y_sorted), bcut_all, nh_all, nb_all, repeat(name), repeat(sketch_type)))
    # elif args.lookup_data:
    #     results = pool.starmap(run_ccm_lookup, zip(repeat(x_valid), repeat(y_valid), nh_all, n_cm_all, repeat(lookup_dict), scut_all, repeat(name), repeat(sketch_type)))
    # else:
    # f = open('output.txt','w')
    # sys.stdout = f
    
    # sys.stdout = sys.stdout
    # f.close()
    # valid_results = np.reshape(valid_results, rshape)
    # space_actual = np.reshape(space_actual, rshape)
    # bcut_all = np.reshape(bcut_all, rshape)
    # scut_all = np.reshape(scut_all, rshape)
    # nh_all = np.reshape(nh_all, rshape)
    # nb_all = np.reshape(nb_all, rshape)
    
    # for i, val in enumerate(valid_results):
    #     print("Error: " + str(val) + " for bucket: " + str(prev_bcut[i]) + " to " + str(next_bcut[i]))
    #     print("Space: " + str(space_actual[i]))
    #     print("=====================================")
    # log_str += '==== valid_results ====\n'
    # for i in range(len(valid_results)):
    #     log_str += 'space: %.2f\n' % args.space_list[i]
    #     for j in range(len(valid_results[i])):
    #         for k in range(len(valid_results[i, j])):
    #             log_str += '%s: bcut: %d, # hashes %d, # buckets %d - \tloss %.2f\tspace %.1f\n' % \
    #                 (name, bcut_all[i,j,k], nh_all[i,j,k], nb_all[i,j,k], valid_results[i,j,k], space_actual[i,j,k])
    # log_str += 'param search done -- time: %.2f sec\n' % (time.time() - start_t)

    # np.savez(os.path.join(folder, args.save+'_valid'),
    #     command=command,
    #     loss_all=valid_results,
    #     b_cutoffs=bcut_all,
    #     n_hashes=nh_all,
    #     n_buckets=nb_all,
    #     space_list=args.space_list,
    #     space_actual=space_actual,
    #     )

    # log_str += '==== best parameters ====\n'
    # rshape = (len(args.space_list), -1)
    # best_param_idx = np.argmin(valid_results.reshape(rshape), axis=1)
    # best_scuts     = scut_all.reshape(rshape)[np.arange(rshape[0]), best_param_idx]
    # best_bcuts     = bcut_all.reshape(rshape)[np.arange(rshape[0]), best_param_idx]
    # best_n_buckets = nb_all.reshape(rshape)[np.arange(rshape[0]), best_param_idx]
    # best_n_hashes  = nh_all.reshape(rshape)[np.arange(rshape[0]), best_param_idx]
    # best_valid_loss  = valid_results.reshape(rshape)[np.arange(rshape[0]), best_param_idx]
    # best_valid_space = space_actual.reshape(rshape)[np.arange(rshape[0]), best_param_idx]

    # for i in range(len(best_valid_loss)):
    #     log_str += 'space: %.2f, scut %.3f, bcut %d, n_buckets %d, n_hashes %d - \tloss %.2f\tspace %.1f\n' % \
    #         (args.space_list[i], best_scuts[i], best_bcuts[i], best_n_buckets[i], best_n_hashes[i], best_valid_loss[i], best_valid_space[i])

    # # test data using best parameters
    # pool = Pool(args.n_workers)
    # if args.perfect_order:
    #     # version 2
    #     y_sorted = np.sort(y_test)[::-1]
    #     results = pool.starmap(run_ccm, zip(repeat(y_sorted), best_bcuts, best_n_hashes, best_n_buckets, repeat(name), repeat(sketch_type)))
    # elif args.lookup_data:
    #     results = pool.starmap(run_ccm_lookup,
    #         zip(repeat(x_test), repeat(y_test), best_n_hashes, best_n_buckets - best_bcuts, repeat(lookup_dict), best_scuts, repeat(name), repeat(sketch_type)))
    # else:
    #     results = pool.starmap(run_ccm_wscore,
    #         zip(repeat(y_test_ordered), repeat(test_scores), best_scuts, best_n_buckets - best_bcuts, best_n_hashes, repeat(name), repeat(sketch_type)))
    # pool.close()
    # pool.join()

    # test_results, space_test = zip(*results)