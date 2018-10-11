import argparse
import sys
import numpy as np
sys.path.append('.')

def softmax(raw_score, T=1):
    exp_s = np.exp((raw_score - raw_score.max(axis=-1)[..., None])*T)
    sum_s = exp_s.sum(axis=-1)
    return exp_s / sum_s[..., None]

def default_aggregation_func(score_arr, normalization=True, crop_agg=None):
    """
    This is the default function for make video-level prediction
    :param score_arr: a 3-dim array with (frame, crop, class) layout
    :return:
    """
    crop_agg = np.mean if crop_agg is None else crop_agg
    if normalization:
        return softmax(crop_agg(score_arr, axis=1).mean(axis=0))
    else:
        return crop_agg(score_arr, axis=1).mean(axis=0)
# from pyActionRecog.utils.metrics import mean_class_accuracy

# parser = argparse.ArgumentParser()
# parser.add_argument('score_files', nargs='+', type=str)
# parser.add_argument('--score_weights', nargs='+', type=float, default=None)
# parser.add_argument('--crop_agg', type=str, choices=['max', 'mean'], default='mean')
# args = parser.parse_args()
def eval_scores(score_files,score_weights = [1.5,1],crop_agg = 'mean'):
    score_npz_files = [np.load(x) for x in score_files]
    # score_weights = args.score_weights
    if len(score_weights) != len(score_npz_files):
        raise ValueError("Only {} weight specifed for a total of {} score files"
                             .format(len(score_weights), len(score_npz_files)))

    score_list = [x['scores'][:, 0] for x in score_npz_files]
    # label_list = [x['labels'] for x in score_npz_files]

    # label verification

    # score_aggregation
    agg_score_list = []
    for score_vec in score_list:
        agg_score_vec = [default_aggregation_func(x, normalization=False, crop_agg=getattr(np,crop_agg)) for x in score_vec]
        agg_score_list.append(np.array(agg_score_vec))

    final_scores = np.zeros_like(agg_score_list[0])
    for i, agg_score in enumerate(agg_score_list):
        final_scores += agg_score * score_weights[i]
    print(final_scores)
    # accuracy
    # acc = mean_class_accuracy(final_scores, label_list[0])
    # print 'Final accuracy {:02f}%'.format(acc * 100)

eval_scores(['/data1/zhuhuihui/zhh/tsn_zhh/test_tsn/score_file_flow_2.npz','/data1/zhuhuihui/zhh/tsn_zhh/test_tsn/score_file_rgb_2.npz'])