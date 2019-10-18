import numpy as np

reward_file = 'rewards_1_2200.txt'
data = np.loadtxt(reward_file, dtype=float)

averaged_items = list()
averages = list()
sum = 0
line_idx = 0
for record in data:
	if line_idx >= 100:
		avg = sum / 100
		averages.append(avg)
		sum -= averaged_items[0]
		averaged_items = averaged_items[1:]
	averaged_items.append(record)
	sum += record
	line_idx += 1

np.savetxt('averages_1_2100.txt', averages)