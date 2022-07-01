
import numpy as np
from skimage.measure import label
from dpipe.im.metrics import dice_score

# ### FROC ###


def volume2diameter(volume):
    return (6 * volume / np.pi) ** (1 / 3)


def get_intersection_stat_dice_id(cc_mask, one_cc, pred=None, logit=None):
    """Returns max local dice and corresponding stat to this hit component.
    If ``pred`` is ``None``, ``cc_mask`` treated as ground truth and stat sets to be 1."""
    hit_components = np.unique(cc_mask[one_cc])
    hit_components = hit_components[hit_components != 0]

    hit_stats = dict(zip(['hit_max', 'hit_median', 'hit_q95', 'hit_logit'], [[], [], [], []]))
    hit_dice, hit_id = [], []

    for n in hit_components:
        cc_mask_hit_one = cc_mask == n
        hit_dice.append(dice_score(cc_mask_hit_one, one_cc))
        hit_id.append(n)

        hit_stats['hit_max'].append(1. if pred is None else np.max(pred[cc_mask_hit_one]))
        hit_stats['hit_median'].append(1. if pred is None else np.median(pred[cc_mask_hit_one]))
        hit_stats['hit_q95'].append(1. if pred is None else np.percentile(pred[cc_mask_hit_one].astype(int), q=95))
        hit_stats['hit_logit'].append(np.inf if logit is None else np.max(logit[cc_mask_hit_one]))

    if len(hit_dice) == 0:
        return dict(zip(['hit_max', 'hit_median', 'hit_q95', 'hit_logit'], [0., 0., 0., -np.inf])), 0., None
    else:
        max_idx = np.argmax(hit_dice)
        hit_id = np.array(hit_id)[max_idx]
        hit_stats['hit_max'] = np.array(hit_stats['hit_max'])[max_idx]
        hit_stats['hit_median'] = np.array(hit_stats['hit_median'])[max_idx]
        hit_stats['hit_q95'] = np.array(hit_stats['hit_q95'])[max_idx]
        hit_stats['hit_logit'] = np.array(hit_stats['hit_logit'])[max_idx]
        return hit_stats, np.max(hit_dice), hit_id


def froc_records(segm, pred, logit):
    segm_split, segm_n_splits = label(segm > 0.5, return_num=True)
    pred_split, pred_n_splits = label(pred > 0.5, return_num=True)

    records = []

    for n in range(1, segm_n_splits + 1):
        record = {}
        segm_cc = segm_split == n

        record['obj'] = f'tum_{n}'
        record['is_tum'] = True
        record['diameter'] = volume2diameter(np.sum(segm_cc))
        stats, dice, hit_id = get_intersection_stat_dice_id(cc_mask=pred_split, one_cc=segm_cc,
                                                            pred=pred, logit=logit)
        record['hit_dice'] = dice
        record['hit_max'], record['hit_median'], record['hit_q95'], record['hit_logit'] = stats.values()
        record['hit_stat'] = record['hit_max']  # backward compatibility
        record['hit_obj'] = f'pred_{hit_id}'
        record['self_stat'] = 1.
        record['self_logit'] = np.inf

        records.append(record)

    for n in range(1, pred_n_splits + 1):
        record = {}
        pred_cc = pred_split == n

        record['obj'] = f'pred_{n}'
        record['is_tum'] = False
        record['diameter'] = volume2diameter(np.sum(pred_cc))
        stats, dice, hit_id = get_intersection_stat_dice_id(cc_mask=segm_split, one_cc=pred_cc)
        record['hit_dice'] = dice
        record['hit_max'], record['hit_median'], record['hit_q95'], record['hit_logit'] = stats.values()
        record['hit_stat'] = record['hit_max']  # backward compatibility
        record['hit_obj'] = f'tum_{hit_id}'
        record['self_stat'] = np.max(pred[pred_cc])
        record['self_logit'] = np.max(logit[pred_cc])

        records.append(record)

    return records
