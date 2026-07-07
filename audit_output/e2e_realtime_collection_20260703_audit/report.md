# Recording Dataset Audit

- root: `/data/hdt/newtrackvla/audit_output/e2e_realtime_collection_20260703`
- episodes: 1
- steps per agent: 200
- dt distribution: `{'0.1': 200}`
- effective_dt distribution: `{'0.076014': 1, '0.076039': 1, '0.076076': 3, '0.076079': 1, '0.076087': 1, '0.076091': 3, '0.076094': 1, '0.076095': 4, '0.083846': 3, '0.08385': 1, '0.083852': 1, '0.099994': 1, '0.099996': 1, '0.099997': 2, '0.100006': 1, '0.100009': 1, '0.108967': 1, '0.109419': 1, '0.109421': 1, '0.109423': 1, '0.109425': 1, '0.109426': 1, '0.109428': 1, '0.125573': 2, '0.125575': 1, '0.125578': 1, '0.12558': 1, '0.131473': 1, '0.132275': 1, '0.132956': 1, '0.133105': 1, '0.133147': 1, '0.133184': 1, '0.133188': 1, '0.133218': 1, '0.133226': 1, '0.133242': 1, '0.133255': 1, '0.133277': 2, '0.133297': 1, '0.133299': 1, '0.133301': 1, '0.133302': 1, '0.13331': 1, '0.133313': 2, '0.133317': 1, '0.133318': 1, '0.133323': 1, '0.133324': 8, '0.133325': 1, '0.133326': 1, '0.133327': 6, '0.133328': 2, '0.133329': 3, '0.13333': 32, '0.133331': 5, '0.133332': 7, '0.133333': 4, '0.133334': 15, '0.133335': 3, '0.133336': 20, '0.133337': 2, '0.133338': 1, '0.133339': 4, '0.13334': 17, '0.133341': 2, '0.133342': 3, '0.133343': 1, '0.133346': 1, '0.166676': 1}`
- ue_interval_ms distribution: `{1000: 200}`
- grade counts: `{'A': 1}`

## Timing

- step_wall_time_s mean/P50/P95/max: 0.0815936 / 0.0776489 / 0.113312 / 0.15128
- target pose skew mean/P95/max: 0 / 0 / 0 m

## Pose Drift
- drone xy drift mean/P50/P95/max: nan / nan / nan / nan m
- drone yaw drift mean/P95/max: nan / nan / nan deg
- robotdog xy drift mean/P50/P95/max: nan / nan / nan / nan m
- robotdog yaw drift mean/P95/max: nan / nan / nan deg

## Action Difference
- drone |base-commanded| mean/P95/max: 0.306178 / 0.869308 / 1.82867
- drone base vs commanded xy angle P50/P95: 2.85477 / 100.664 deg
- robotdog |base-commanded| mean/P95/max: 0.318463 / 0.699948 / 1.08619
- robotdog base vs commanded xy angle P50/P95: 1.26012 / 2.04022 deg

## Visibility
- drone visible mean/P50/P95: 1 / 1 / 1
- drone distance mean/P95/max: 4.24954 / 4.95067 / 5.03371 m
- robotdog visible mean/P50/P95: 1 / 1 / 1
- robotdog distance mean/P95/max: 5.94927 / 6.59927 / 6.66267 m
- visible combo distribution: `{'DR': 200}`

See CSV files for per-episode and per-step evidence.
