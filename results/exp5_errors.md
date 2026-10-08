# Experiment 5 — Error Analysis

**Test accuracy:** 0.7843

**Test macro F1:** 0.7838

**Total errors:** 1,838 / 8,522 (21.57%)

## Per-class metrics

```
                    precision    recall  f1-score   support

          business     0.7008    0.6514    0.6752       852
     entertainment     0.8015    0.7629    0.7817       852
        food_drink     0.8253    0.8592    0.8419       852
   health_wellness     0.6708    0.7046    0.6872       853
         parenting     0.7888    0.8239    0.8060       852
          politics     0.7794    0.8087    0.7938       852
science_technology     0.7471    0.6893    0.7171       853
            sports     0.8545    0.8615    0.8580       852
      style_beauty     0.8873    0.8685    0.8778       852
            travel     0.7857    0.8134    0.7993       852

          accuracy                         0.7843      8522
         macro avg     0.7841    0.7843    0.7838      8522
      weighted avg     0.7841    0.7843    0.7838      8522

```

## Top confusion pairs

|                                           |   count |
|:------------------------------------------|--------:|
| ('science_technology', 'business')        |      70 |
| ('business', 'science_technology')        |      61 |
| ('science_technology', 'health_wellness') |      60 |
| ('business', 'politics')                  |      59 |
| ('business', 'health_wellness')           |      58 |
| ('parenting', 'health_wellness')          |      54 |
| ('health_wellness', 'business')           |      52 |
| ('health_wellness', 'parenting')          |      49 |
| ('entertainment', 'politics')             |      47 |
| ('politics', 'business')                  |      45 |

## Confusion matrix (rows=true, cols=pred)

|                    |   business |   entertainment |   food_drink |   health_wellness |   parenting |   politics |   science_technology |   sports |   style_beauty |   travel |
|:-------------------|-----------:|----------------:|-------------:|------------------:|------------:|-----------:|---------------------:|---------:|---------------:|---------:|
| business           |        555 |              13 |           26 |                58 |          27 |         59 |                   61 |       13 |             10 |       30 |
| entertainment      |         11 |             650 |            9 |                16 |          28 |         47 |                   19 |       37 |             21 |       14 |
| food_drink         |          9 |              10 |          732 |                24 |          15 |          3 |                    9 |        4 |             11 |       35 |
| health_wellness    |         52 |              17 |           35 |               601 |          49 |         22 |                   33 |       16 |             10 |       18 |
| parenting          |         12 |              14 |           12 |                54 |         702 |          6 |                   16 |        9 |             15 |       12 |
| politics           |         45 |              25 |            3 |                21 |          13 |        689 |                   18 |       15 |              3 |       20 |
| science_technology |         70 |              19 |           13 |                60 |          29 |         23 |                  588 |       13 |             10 |       28 |
| sports             |         11 |              32 |            6 |                11 |           7 |         22 |                   11 |      734 |              5 |       13 |
| style_beauty       |          4 |              23 |           11 |                30 |           7 |          2 |                   11 |        5 |            740 |       19 |
| travel             |         23 |               8 |           40 |                21 |          13 |         11 |                   21 |       13 |              9 |      693 |