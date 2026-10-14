"""Paired-fold means and Rgamma_mean - R0_mean. Numbers only."""
import csv
import os

BASE = os.path.dirname(os.path.abspath(__file__))
R0_CSV = os.path.join(BASE, 'metrics_R0.csv')
RG_CSV = os.path.join(BASE, 'metrics_Rgamma.csv')
OUT = os.path.join(BASE, 'P7_summary.md')
# AUC is the value RANK prints as AUC, stored in the csv as AUCme.
FOLD_COLS = [
    ('AP', 'AP'),
    ('HL', 'HL'),
    ('RL', 'RL'),
    ('AUC', 'AUCme'),
    ('one_error', 'one_error'),
    ('coverage', 'coverage'),
    ('macAUC', 'macAUC'),
    ('macro_f1', 'macro_f1'),
    ('micro_f1', 'micro_f1'),
    ('best_epoch', 'best_epoch'),
]


def _read(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return {}
    rows = {}
    with open(path, newline='') as handle:
        for row in csv.DictReader(handle):
            if not row or row.get('fold_idx') in (None, ''):
                continue
            rows[int(row['fold_idx'])] = row
    return rows


def _mean_sd(vals):
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n
    return mean, var ** 0.5


def _fold_table(rows):
    header = '| fold_idx | ' + ' | '.join(label for label, _ in FOLD_COLS) + ' |'
    sep = '| --- | ' + ' | '.join('---:' for _ in FOLD_COLS) + ' |'
    lines = [header, sep]
    for fold in sorted(rows):
        row = rows[fold]
        cells = []
        for _, key in FOLD_COLS:
            val = float(row[key])
            cells.append('%d' % val if key == 'best_epoch' else '%.6f' % val)
        lines.append('| %d | %s |' % (fold, ' | '.join(cells)))
    if rows:
        stats = []
        for _, key in FOLD_COLS:
            mean, sd = _mean_sd([float(rows[f][key]) for f in rows])
            if key == 'best_epoch':
                stats.append('%.3f±%.3f' % (mean, sd))
            else:
                stats.append('%.6f±%.6f' % (mean, sd))
        lines.append('| mean±sd | %s |' % ' | '.join(stats))
    return lines


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
        'epochs: 250',
        'folds_requested: 10',
        'n_completed_R0: %d' % len(r0),
        'n_completed_Rgamma: %d' % len(rg),
        'paired_fold_idx: %s' % (paired if paired else '[]'),
        'delta: Rgamma_mean - R0_mean',
        'AUC: AUCme',
        'sd: ddof=0',
        '',
        '## R0',
        '',
    ]
    lines.extend(_fold_table(r0) if r0 else ['no_rows'])
    lines.extend(['', '## Rgamma', ''])
    lines.extend(_fold_table(rg) if rg else ['no_rows'])
    lines.extend(['', '## paired', ''])
    if not paired:
        lines.append('no_paired_folds')
    else:
        lines.append('| metric | n | R0_mean | R0_sd | Rgamma_mean | Rgamma_sd | delta_Rgamma_minus_R0 |')
        lines.append('| --- | ---: | ---: | ---: | ---: | ---: | ---: |')
        for label, key in FOLD_COLS:
            m0, s0 = _mean_sd([float(r0[f][key]) for f in paired])
            mg, sg = _mean_sd([float(rg[f][key]) for f in paired])
            if key == 'best_epoch':
                lines.append('| %s | %d | %.3f | %.3f | %.3f | %.3f | %.3f |' % (
                    label, len(paired), m0, s0, mg, sg, mg - m0))
            else:
                lines.append('| %s | %d | %.6f | %.6f | %.6f | %.6f | %.6f |' % (
                    label, len(paired), m0, s0, mg, sg, mg - m0))
    lines.append('')
    text = '\n'.join(lines)
    with open(OUT, 'w') as handle:
        handle.write(text)
    print(text)


if __name__ == '__main__':
    main()
