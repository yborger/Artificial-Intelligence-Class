# Comparing local search methods

You should use the coordinates `North_America_75.json` and do
multiple runs for each local search. Be sure to experiment with the
default parameter settings to try to get the best results you can.
Then run at least 3 experiments using each search method and compute
the average best cost found.

| HC     | Best Cost |
| ------ | --------- |
| run 1  |  1564.89  |
| run 2  |  1291.70  |
| run 3  |  1308.53  |
| Avg    |  1368.37  |

HC parameters: 
1. original
2. "runs":5000,
    "steps":3000,
    "init_temp":10,
    "temp_decay":0.99
3. "runs":1900,
    "steps":100,
    "rand_move_prob": 0.25

| SA     | Best Cost |
| ------ | --------- |
| run 1  |  503.536  |
| run 2  |  461.717  |
| run 3  |  396.62   |
| Avg    |   453.96  |

SA parameters: 
1. original
2. "runs":50,
    "steps":2500,
    "init_temp":50,
    "temp_decay":0.99
3. "runs":5000,
    "steps":3000,
    "init_temp":10,
    "temp_decay":0.99

| BS     | Best Cost |
| ------ | --------- |
| run 1  |  400.84   |
| run 2  |  569.63   |
| run 3  |  348.25   |
| Avg    |  439.57   |

BS parameters: 
1. original 
2. "pop_size":80,
    "steps":80,
    "init_temp":10,
    "temp_decay":0.99,
    "max_neighbors":5
3. "pop_size":100,
    "steps":200,
    "init_temp":10,
    "temp_decay":0.99,
    "max_neighbors":10




Which local search algorithm (HC, SA, or BS) most consistently finds
the best tours and why do you think it outperforms the others?

The consistency of the algorithms directly correlates with how many runs you do. 
We can clearly notice that Hill Climbing is just not the best (sorry). The reason it is not the best is because the random chance restarts do not guarantee finding a better tour option.
The best average of all of the three is Beam Search, though it was very close to the average of Simulated Annealing.
It was more consistent with fewer runs than Simulated Annealing, which makes sense considering the amount of overall loops to verify the best tours.


