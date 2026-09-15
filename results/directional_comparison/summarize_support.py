"""Summarize which recorded directional rows use fallback or unseen-entity fits."""
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent

def main():
    data = pd.read_csv(HERE / 'model_comparison.csv')
    names = ['forecast_pooled_fallback_rows', 'outcome_pooled_fallback_rows',
             'forecast_unseen_entity_rows', 'outcome_unseen_entity_rows']
    rows = []
    for (domain, configuration), group in data.groupby(['domain', 'configuration'], sort=True):
        if not configuration.startswith('directional_'):
            continue
        assert group['evaluation_rows'].nunique() == 1
        row = dict(domain=domain, configuration=configuration, candidates=len(group),
                   evaluation_rows=int(group['evaluation_rows'].iloc[0]))
        for name in names:
            assert group[name].nunique() == 1
            count = int(group[name].iloc[0])
            assert 0 <= count <= row['evaluation_rows']
            row[name] = count
            row[name.replace('_rows', '_fraction')] = count / row['evaluation_rows']
        rows.append(row)
    pd.DataFrame(rows).to_csv(HERE / 'directional_support.csv', index=False)

if __name__ == '__main__':
    main()
