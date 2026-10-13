"""Mechanical means and delta from per-fold CSVs. No interpretation."""
import csv
import os

BASE = os.path.dirname(os.path.abspath(__file__))
METRICS = ['AP', 'HL', 'RL', 'AUCme', 'one_error', 'coverage', 'macAUC', 'macro_f1', 'micro_f1', 'best_epoch']
R0_CSV = os.path.join(BASE, 'metrics_R0.csv')
RG_CSV = os.path.join(BASE, 'metrics_Rgamma.csv')
OUT = os.path.join(BASE, 'P7_summary.md')


def _read(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return {}
    rows = {}
    with open(path, newline='') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row or row.get('fold_idx') in (None, ''):
                continue
            fold = int(row['fold_idx'])
            rows[fold] = row
    return rows


def _mean(rows, folds, key):
    vals = [float(rows[f][key]) for f in folds]
    return sum(vals) / len(vals)


def main():
    r0 = _read(R0_CSV)
    rg = _read(RG_CSV)
    paired = sorted(set(r0) & set(rg))
    lines = [
        '# P7 summary',
        '',
        'dataset: corel5k',
        'mask_view_ratio: 0.5',
        'mask_label_ratio: 0.5',
        'training_sample_ratio: 0.7',
        'folds_requested: 10',
        'epochs: 250',
        'n_completed_R0: %d' % len(r0),
        'n_completed_Rgamma: %d' % len(rg),
        'paired_fold_idx: %s' % (paired if paired else '[]'),
        'delta: Rgamma_mean - R0_mean on paired_fold_idx',
        '',
    ]
    if not paired:
        lines.append('status: no_paired_folds')
        lines.append('')
    else:
        lines.append('status: %s' % ('complete' if paired == list(range(10)) and len(r0) == 10 and len(rg) == 10 else 'partial'))
        lines.append('')
        lines.append('| metric | n | R0_mean | Rgamma_mean | delta_Rgamma_minus_R0 |')
        lines.append('| --- | ---: | ---: | ---: | ---: |')
        for key in METRICS:
            m0 = _mean(r0, paired, key)
            mg = _mean(rg, paired, key)
            lines.append('| %s | %d | %.10f | %.10f | %.10f |' % (key, len(paired), m0, mg, mg - m0))
        lines.append('')
    with open(OUT, 'w') as handle:
        handle.write('\n'.join(lines))
    print('wrote', OUT)


if __name__ == '__main__':
    main()
