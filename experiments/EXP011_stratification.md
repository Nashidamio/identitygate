# EXP011 - DEV POR stratified by GT attribute

Overall: 190 events, POR_w30 = 0.8579


## Synchronized whole-scene occlusion

| sync_video   |   events |   POR |   videos |
|:-------------|---------:|------:|---------:|
| False        |      108 | 0.769 |       36 |
| True         |       82 | 0.976 |        4 |


## Occlusion gap length (frames)

| gap_len   |   events |   POR |   videos |
|:----------|---------:|------:|---------:|
| 5-9       |       42 | 0.81  |       21 |
| 10-19     |       32 | 0.75  |       20 |
| 20-39     |       23 | 0.696 |       15 |
| 40+       |       93 | 0.957 |       11 |


## Object size (mean GT area fraction when visible)

| obj_size   |   events |   POR |   videos |
|:-----------|---------:|------:|---------:|
| <0.5%      |       68 | 0.809 |       17 |
| 0.5-2%     |       78 | 0.897 |       16 |
| 2-5%       |       27 | 0.926 |       10 |
| >5%        |       17 | 0.765 |        9 |


## Video length (frames)

| n_frames   |   events |   POR |   videos |
|:-----------|---------:|------:|---------:|
| 60-80      |       12 | 0.75  |        8 |
| 81-120     |       90 | 0.889 |       19 |
| 121-200    |       54 | 0.926 |        7 |
| 200+       |       34 | 0.706 |        6 |


## Distractor density (mean objects visible per frame)

| mean_density   |   events |   POR |   videos |
|:---------------|---------:|------:|---------:|
| <=2            |       64 | 0.766 |       17 |
| 2-4            |       50 | 0.94  |       11 |
| 4-8            |       56 | 0.839 |       11 |
| 8+             |       20 | 1     |        1 |
