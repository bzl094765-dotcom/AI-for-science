"""Check R0 matches the official StandardScaler-then-mask line, and Rγ matches L1 -> x**0.25 -> L2 then the same mask."""
import os
import sys

import numpy as np
import scipy.io
from sklearn.preprocessing import StandardScaler

WORKDIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(WORKDIR, 'workdir_rank'))
import MLdataset  # noqa: E402

MAT = os.path.join(WORKDIR, 'workdir_rank', 'data', 'corel5k', 'corel5k_six_view.mat')
FOLD = os.path.join(
    WORKDIR, 'workdir_rank', 'data', 'corel5k',
    'corel5k_six_view_MaskRatios_0.5_LabelMaskRatio_0.5_TraindataRatio_0.7.mat')


def _prepared(fold_idx=0):
    data = scipy.io.loadmat(MAT)
    datafold = scipy.io.loadmat(FOLD)
    mv_data = data['X'][0]
    labels = data['label'].astype(np.float32)
    if labels.min() == -1:
        labels = (labels + 1) * 0.5
    if labels.shape[0] in mv_data[0].shape:
        total_sample_num = labels.shape[0]
    else:
        total_sample_num = labels.shape[0]
    if total_sample_num != mv_data[0].shape[0]:
        mv_data = [v_data.T for v_data in mv_data]
    if total_sample_num != labels.shape[0]:
        labels = labels.T
    folds_data = datafold['folds_data']
    folds_label = datafold['folds_label']
    folds_sample_index = datafold['folds_sample_index']
    inc_view_indicator = np.array(folds_data[0, fold_idx], 'int32')
    inc_label_indicator = np.array(folds_label[0, fold_idx], 'int32')
    sample_index = np.array(folds_sample_index[0, fold_idx], 'int32').reshape(-1) - 1
    labels = labels[sample_index]
    inc_view_indicator = inc_view_indicator[sample_index]
    inc_label_indicator = inc_label_indicator[sample_index]
    mv_data = [v_data[sample_index, :] for v_data in mv_data]
    names = [str(x.item()) if hasattr(x, 'item') else str(x) for x in data['features'].reshape(-1)]
    return mv_data, inc_view_indicator, labels, names


def _official(mv_data, inc_view_indicator):
    return [
        (StandardScaler().fit_transform(v_data.astype(np.float32)) * inc_view_indicator[:, v:v + 1])
        for v, v_data in enumerate(mv_data)
    ]


def _reference_power(v_data, mask_col):
    x = np.asarray(v_data, dtype=np.float64)
    l1 = np.abs(x).sum(axis=1, keepdims=True)
    np.divide(x, l1, out=x, where=l1 != 0)
    x = x ** 0.25
    l2 = np.sqrt(np.square(x).sum(axis=1, keepdims=True))
    np.divide(x, l2, out=x, where=l2 != 0)
    return (x.astype(np.float32) * mask_col)


def main():
    mv_data, inc_view_indicator, labels, names = _prepared(0)
    dims = [int(v.shape[1]) for v in mv_data]
    print('features', names)
    print('dims', dims, 'n', mv_data[0].shape[0], 'labels', labels.shape)
    if names != ['DenseHue.hvecs', 'DenseSift.hvecs', 'Gist.fvec', 'Hsv.hvecs32', 'Rgb.hvecs32', 'Lab.hvecs32']:
        raise SystemExit('unexpected feature names: %s' % names)
    if dims != [100, 1000, 512, 4096, 4096, 4096]:
        raise SystemExit('unexpected dims: %s' % dims)

    os.environ['RANK_PREPROCESS'] = 'r0'
    MLdataset._transform_views_then_mask._logged = False
    r0 = MLdataset._transform_views_then_mask(mv_data, inc_view_indicator)
    official = _official(mv_data, inc_view_indicator)
    for v, (a, b) in enumerate(zip(r0, official)):
        if a.shape != b.shape or a.dtype != b.dtype or not np.array_equal(a, b):
            raise SystemExit('R0 != official StandardScaler line on view %d' % v)
    print('R0 array_equal official line: ok')

    os.environ['RANK_PREPROCESS'] = 'rgamma'
    MLdataset._transform_views_then_mask._logged = False
    rg = MLdataset._transform_views_then_mask(mv_data, inc_view_indicator)
    if not np.array_equal(rg[2], official[2]):
        raise SystemExit('Gist Rγ != Gist R0/official')
    print('Gist Rγ array_equal official StandardScaler: ok')
    for v in (0, 1, 3, 4, 5):
        ref = _reference_power(mv_data[v], inc_view_indicator[:, v:v + 1])
        if not np.array_equal(rg[v], ref):
            max_abs = np.max(np.abs(rg[v].astype(np.float64) - ref.astype(np.float64)))
            raise SystemExit('Rγ power mismatch view %d max_abs %s' % (v, max_abs))
        if np.array_equal(rg[v], official[v]):
            raise SystemExit('Rγ view %d unexpectedly equal to StandardScaler' % v)
        observed = inc_view_indicator[:, v] == 1
        rows = rg[v][observed]
        norms = np.linalg.norm(rows.astype(np.float64), axis=1)
        if not np.allclose(norms, 1.0, atol=1e-5):
            raise SystemExit('observed-row L2 not 1 on view %d' % v)
        missing = inc_view_indicator[:, v] == 0
        if not np.all(rg[v][missing] == 0):
            raise SystemExit('missing rows not masked to 0 on view %d' % v)
    print('Rγ L1->**0.25->L2 then mask: ok')
    loaded_r0 = MLdataset.loadMfDIMvMlDataFromMat(MAT, FOLD, fold_idx=0)[0]
    os.environ['RANK_PREPROCESS'] = 'r0'
    # loader reads the env at call time; force r0 again for the returned object check
    loaded_r0 = MLdataset.loadMfDIMvMlDataFromMat(MAT, FOLD, fold_idx=0)[0]
    for v in range(6):
        if not np.array_equal(loaded_r0[v], official[v]):
            raise SystemExit('loadMfDIMvMlDataFromMat R0 mismatch view %d' % v)
    print('loadMfDIMvMlDataFromMat R0: ok')


if __name__ == '__main__':
    main()
