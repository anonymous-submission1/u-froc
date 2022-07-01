import argparse
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm
from skimage.measure import label
from lazycon import load as read_config
from dpipe.im.metrics import dice_score
from dpipe.io import load, save

from ufroc.utils import np_sigmoid
from ufroc.paths import EXP_BASE_PATH


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, type=str,
                        choices=('met', 'lidc', 'lits', 'egd', 'midrc', 'lits_mod'))
    parser.add_argument('--single', required=False, action='store_true', default=False)
    args = parser.parse_known_args()[0]

    # ### args ###
    n_repeats = 5
    eps = 1e-9
    stat_aggregation_fns_dict = {'mean': np.mean, 'median': np.median,
                                 'min': np.min, 'q5': lambda x: np.percentile(x, 5),
                                 'max': np.max, 'q95': lambda x: np.percentile(x, 95),
                                 'sum_log': lambda x: np.sum(np.log(x + eps))}

    # proba_ths = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])  # looks like FP reduction...
    # proba_ths = 1.5 - np.logspace(1, 0, num=7, endpoint=False, base=2) / 2  # alternatively, but it is similar

    proba_ths = np.array([0.25, 0.5, 0.75])

    # ### paths ###
    base_path = Path(EXP_BASE_PATH)
    exp_paths = [base_path / f'{args.dataset}_{n}' for n in range(n_repeats)]
    res_path = (base_path / f'{args.dataset}_froc_single') if args.single else (base_path / f'{args.dataset}_froc')
    res_path.mkdir(exist_ok=True)
    froc_file_postfixes = ['true', 'pred', 'hit']

    # ### config stuff ###
    config = read_config(exp_paths[0] / 'resources.config')
    load_y = config.load_y
    if args.dataset in ('egd', 'midrc', 'lits_mod'):  # empty masks
        load_y = lambda i: np.zeros_like(config.load_x(i))

    # ### ids ###
    ids = load(exp_paths[0] / 'experiment_0/test_ids.json')

    for _id in tqdm(ids):

        # ### check ? skip ###
        froc_files = [res_path / f'{_id}_{postfix}.json' for postfix in froc_file_postfixes]
        if all(map(lambda p: p.exists(), froc_files)):
            continue

        # ### load logits and ground truth ###
        logits = [load(p / f'experiment_0/test_predictions/{_id}.npy') for p in exp_paths]
        y = load_y(_id)

        # ### prediction ###
        preds = [np_sigmoid(logit) for logit in logits]
        avg_logit = np.mean(logits, axis=0)
        avg_pred = np_sigmoid(avg_logit)

        # ### defining stat maps ###
        stat_map_logit_1 = logits[0]
        stat_map_logit_n = avg_logit
        stat_map_pred_1 = preds[0]
        stat_map_pred_n = avg_pred
        stat_map_entropy_1 = _entropy(preds=None, avg_pred=preds[0])
        stat_map_entropy_n = _entropy(preds=None, avg_pred=avg_pred)
        stat_map_avg_entropy = _avg_entropy(preds=preds)
        stat_map_dispersion = _dispersion(logits)
        stat_map_mutual_info = _mutual_info(preds=None, entropy=stat_map_entropy_n, avg_entropy=stat_map_avg_entropy)
        stat_map_dices = preds  # list !

        stat_maps_dict = {
            'logit_1': stat_map_logit_1,
            'logit_n': stat_map_logit_n,
            'pred_1': stat_map_pred_1,
            'pred_n': stat_map_pred_n,
            'entropy_1': stat_map_entropy_1,
            'entropy_n': stat_map_entropy_n,
            'avg_entropy': stat_map_avg_entropy,
            'dispersion': stat_map_dispersion,
            'mutual_info': stat_map_mutual_info,
            'obj_dsc': stat_map_dices,  # list !
        }

        pred = preds[0] if args.single else avg_pred

        froc_records = get_froc_records(true=y, pred=pred, proba_ths=proba_ths, stat_maps_dict=stat_maps_dict,
                                        stat_aggregation_fns_dict=stat_aggregation_fns_dict, _id=_id)

        for postfix, froc_file in zip(froc_file_postfixes, froc_files):
            save(froc_records[postfix], froc_file)


def get_froc_records(true, pred, proba_ths, stat_maps_dict, stat_aggregation_fns_dict, _id):
    records = defaultdict(list)

    # ### records true: ###
    true_split, true_n_splits = label(true > 0.5, connectivity=3, return_num=True)
    for lbl_i in range(1, true_n_splits + 1):
        records['true'].append({'ID': _id, 'obj_id': f'true_{lbl_i}'})

    for th in proba_ths:

        pred_split, pred_n_splits = label(pred > th, connectivity=3, return_num=True)
        for lbl_i in range(1, pred_n_splits + 1):

            pred_cc = pred_split == lbl_i
            pred_id = f'pred_{lbl_i}'

            # ### records hit: ###
            for record in get_hit_records(pred_cc=pred_cc, true_ccs=true_split, pred_id=pred_id, _id=_id, proba_th=th):
                records['hit'].append(record)

            # ### records pred: ###
            records['pred'].append(get_pred_record(pred_cc=pred_cc, stat_maps_dict=stat_maps_dict,
                                                   stat_aggregation_fns_dict=stat_aggregation_fns_dict,
                                                   pred_id=pred_id, _id=_id, proba_th=th))

    return records


def get_hit_records(pred_cc, true_ccs, pred_id, _id, proba_th):
    hit_cc_ids = np.unique(true_ccs[pred_cc])
    hit_cc_ids = hit_cc_ids[hit_cc_ids != 0]

    hit_records = []
    for hit_id in hit_cc_ids:
        hit_records.append({'ID': _id, 'proba_th': proba_th, 'obj_id': pred_id, 'hit_id': f'true_{hit_id}',
                            'DSC': dice_score(pred_cc, true_ccs == hit_id), })

    return hit_records


def get_pred_record(pred_cc, stat_maps_dict, stat_aggregation_fns_dict, pred_id, _id, proba_th):
    pred_records = {'ID': _id, 'proba_th': proba_th, 'obj_id': pred_id, }

    for stat_name, stat_map in stat_maps_dict.items():
        if stat_name == 'obj_dsc':  # and isinstance(stat_map, list):
            cc_stats = []
            for _pred in stat_map:
                hit_records = get_hit_records(pred_cc=pred_cc, true_ccs=label(_pred > proba_th, connectivity=3),
                                              pred_id=pred_id, _id=_id, proba_th=proba_th)
                cc_stats.append(0 if len(hit_records) == 0 else max(hit_records, key=lambda x: x['DSC'])['DSC'])
            cc_stats = np.array(cc_stats)
        else:
            cc_stats = stat_map[pred_cc]

        for aggr_name, aggr_fn in stat_aggregation_fns_dict.items():
            warnings.filterwarnings('ignore')
            pred_records[f'{stat_name}|{aggr_name}'] = aggr_fn(cc_stats)
            warnings.filterwarnings('default')

    return pred_records


# ### ### Statistics ### ###


def _dispersion(logits):
    return np.var(logits, axis=0)


def _entropy(preds, avg_pred=None, eps=1e-9):
    if avg_pred is None:
        avg_pred = np.mean(preds, axis=0)

    warnings.filterwarnings('ignore')
    entropy = - (avg_pred * np.log2(avg_pred + eps) + (1 - avg_pred) * np.log2(1 - avg_pred + eps))
    warnings.filterwarnings('default')

    entropy[avg_pred == 0] = 0
    entropy[avg_pred == 1] = 0

    return entropy


def _avg_entropy(preds, eps=1e-9):
    preds = np.asarray(preds)
    warnings.filterwarnings('ignore')
    entropies = - (preds * np.log2(preds + eps) + (1 - preds) * np.log2(1 - preds + eps))
    warnings.filterwarnings('default')

    entropies[preds == 0] = 0
    entropies[preds == 1] = 0

    return np.mean(entropies, axis=0)


def _mutual_info(preds, avg_pred=None, entropy=None, avg_entropy=None, eps=1e-9):
    if entropy is None:
        if avg_pred is None:
            avg_pred = np.mean(preds, axis=0)
        entropy = _entropy(preds=None, avg_pred=avg_pred, eps=eps)

    if avg_entropy is None:
        avg_entropy = _avg_entropy(preds=preds, eps=eps)
    return entropy - avg_entropy


# ### ### Entrypoint ### ###


if __name__ == '__main__':
    main()
