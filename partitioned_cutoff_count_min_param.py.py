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
from sketch import count_min_partitioned, count_sketch

k = 6
rangeMax = 3129.52612
rangeMin = -99.1813681 # Will have to round up negative values here

def cutoff_countmin_p(y, scores, score_start, score_end, n_cm_buckets, n_hashes):
    """ Learned Partitioned Count-Min (use predicted scores to identify heavy hitters)
        Args:
        y: true counts of each item (sorted, largest first), float - [num_items]
        scores: predicted scores of each item - [num_items]
        score_cutoff: threshold for heavy hitters
        n_cm_buckets: number of buckets of Count-Min
        n_hashes: number of hash functions

    Returns:
        loss_avg: estimation error
        space: space usage in bytes
    """
    if len(y) == 0:
        return 0            # avoid division of 0

    if score_start == 0: score_start = -99.1813681
    y_cm = []
    for val in y:
        if val >= score_start and val < score_end:
            y_cm.append(val)
    # print(y_cm)
    loss_count, count = count_min_partitioned(y_cm, n_cm_buckets, n_hashes)
    # print('loss_micro_sketch %.2f\t' % (loss_cm))

    space = n_cm_buckets * n_hashes * 4
    return loss_count, count, space

def run_ccm_p(y, scores, score_start, score_end, n_cm_buckets, n_hashes, name):
    start_t = time.time()
    # if sketch_type == 'count_min':
    loss, sumCount, space = cutoff_countmin_p(y, scores, score_start, score_end, n_cm_buckets, n_hashes)
    # else:
    # I THINK LOSS IS THE ERROR HERE
    #     loss, space = cutoff_countsketch_wscore(y, scores, score_cutoff, n_cm_buckets, n_hashes)
    # f = open('filename.txt', 'a')
    # print('thresholdstart:%s, thresholdend:%s, # hashes %s, # space: %s # cm buckets %s - loss %s time: %s sec ' % \
    # (str(score_start.round(2)), str(score_end.round(2)), str(n_hashes), str(space), str(n_cm_buckets), str(loss / sumCount if sumCount != 0 else 0), str(time.time() - start_t)), file=f)
    # f.close()
    return loss, sumCount, space

# def run_ccm_lookup(x, y, n_hashes, n_cm_buckets, d_lookup, s_cutoff, name, sketch_type):
#     start_t = time.time()
#     if sketch_type == 'count_min':
#         loss, space = cutoff_lookup(x, y, n_cm_buckets, n_hashes, d_lookup, s_cutoff)
#     else:
#         loss, space = cutoff_lookup(x, y, n_cm_buckets, n_hashes, d_lookup, s_cutoff, \
#             sketch='CountSketch')
#     print('%s: s_cut: %d, # hashes %d, # cm buckets %d - loss %.2f\t time: %.2f sec' % \
#         (name, s_cutoff, n_hashes, n_cm_buckets, loss, time.time() - start_t))
#     return loss, space

# def get_great_cut(b_cut, y, max_bcut):
#     assert b_cut <= max_bcut
#     y_sorted = np.sort(y)[::-1]
#     if b_cut < len(y_sorted):
#         s_cut = y_sorted[b_cut]
#     else:
#         s_cut = y_sorted[-1]

#     # return cut at the boundary of two frequencies
#     n_after_same = np.argmax((y_sorted == s_cut)[::-1]) # items after items == s_cut
#     if (len(y) - n_after_same) < max_bcut:
#         b_cut_new = (len(y) - n_after_same)
#         if n_after_same == 0:
#             s_cut = s_cut - 1   # get every thing
#         else:
#             s_cut = y_sorted[b_cut_new] # item right after items == s_cut
#     else:
#         b_cut_new = np.argmax(y_sorted == s_cut) # first item that # items == s_cut
#     return b_cut_new, s_cut


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
    args = argparser.parse_args()

    assert not (args.perfect_order and args.lookup_data),   "use either --perfect or --lookup"

    command = ' '.join(sys.argv) + '\n'
    log_str = command
    log_str += git_log() + '\n'
    print(log_str)
    np.random.seed(args.seed)

    if args.count_sketch:
        sketch_type = 'count_sketch'
    else:
        sketch_type = 'count_min'

    name = "partitioned_count_min_sketch"

    folder = os.path.join('param_results', name, '')
    if not os.path.exists(folder):
        os.makedirs(folder)

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
    f = open('filename.txt', 'a')
    for space in args.space_list:
        # Maintain this constant space 
        # Naively create cuts according to log bucket partitioning 
        b_cutoffs = np.logspace(0, np.log10(rangeMax), 6, 2)
        prev_bcut =  b_cutoffs[0:len(b_cutoffs)-1]
        prev_bcut = np.insert(prev_bcut,0, 0)
        next_bcut = b_cutoffs
        for n_hash in args.n_hashes_list:
            n_cmin_buckets = int(space * 1e6 / (n_hash * k)) 
            # Naive space per sketch: every bucket gets equal space 
            # n_cmin_buckets = int((space * 1e6 - bcut * 4 * cutoff_cost_mul) / (n_hash * 4))
            # nh_all.append(n_hash)
            # nb_all.append(n_cmin_buckets)
            # finPrev = np.concatenate((finPrev, prev_bcut))
            # finNext = np.concatenate((finNext, next_bcut))
            start_t = time.time()
            pool = Pool(args.n_workers)
            results = pool.starmap(run_ccm_p, zip(repeat(y_valid_ordered), repeat(test_scores), prev_bcut, next_bcut, repeat(n_cmin_buckets), repeat(n_hash), repeat(name)))
            pool.close()
            pool.join()
            loss_results, sumResults, space_actual = zip(*results)
            bigE = np.sum(loss_results) / np.sum(sumResults)
            print("Space: " + str(space) + " n_hash: " + str(n_hash), file=f)
            print("====================================", file=f)
            print("Big error: " + str(bigE), file=f)
            print("====================================", file=f)
            val, num = count_min_partitioned(y_valid_ordered, int(space * 1e6 / (n_hash)) , n_hash)
            print("Count-min sketch error rate: " + str(val/num),file=f)
            print("====================================", file=f)
    f.close()

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
    # # for i in range(len(valid_results)):
    # #     log_str += 'space: %.2f\n' % args.space_list[i]
    # #     for j in range(len(valid_results[i])):
    # #         for k in range(len(valid_results[i, j])):
    # #             log_str += '%s: bcut: %d, # hashes %d, # buckets %d - \tloss %.2f\tspace %.1f\n' % \
    # #                 (name, bcut_all[i,j,k], nh_all[i,j,k], nb_all[i,j,k], valid_results[i,j,k], space_actual[i,j,k])
    # # log_str += 'param search done -- time: %.2f sec\n' % (time.time() - start_t)

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

    # log_str += '==== test test_results ====\n'
    # for i in range(len(test_results)):
    #     log_str += 'space: %.2f, scut %.3f, bcut %d, n_buckets %d, n_hashes %d - \tloss %.2f\tspace %.1f\n' % \
    #            (args.space_list[i], best_scuts[i], best_bcuts[i], best_n_buckets[i], best_n_hashes[i], test_results[i], space_test[i])

    # log_str += 'total time: %.2f sec\n' % (time.time() - start_t)
    # print(log_str)
    # with open(os.path.join(folder, args.save+'.log'), 'w') as f:
    #     f.write(log_str)

    # np.savez(os.path.join(folder, args.save+'_test'),
    #     command=command,
    #     loss_all=test_results,
    #     s_cutoffs=best_scuts,
    #     b_cutoffs=best_bcuts,
    #     n_hashes=best_n_hashes,
    #     n_buckets=best_n_buckets,
    #     space_list=args.space_list,
    #     space_actual=space_test,
    #     )