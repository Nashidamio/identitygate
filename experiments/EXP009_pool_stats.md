# EXP009 pre-flight — anchor policy cost

Source: experiments/EXP006_events.csv (no new computation).

## Pool survival under frame-0 anchor requirement

| filter                                             |   videos |   tracks |   events |
|:---------------------------------------------------|---------:|---------:|---------:|
| provisional pool (all tracks)                      |      291 |      743 |     1023 |
| (a) event-track visible at frame 0                 |      291 |      743 |     1023 |
| (a-strict) ALL objects in video visible at frame 0 |      291 |      743 |     1023 |


## First-visible frame distribution (pool tracks)

|   first_visible == 0 |   1..9 |   10..29 |   30+ |
|---------------------:|-------:|---------:|------:|
|                  743 |      0 |        0 |     0 |

