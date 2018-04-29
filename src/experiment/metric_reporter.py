import matplotlib.pyplot as plt
import numpy as np
import dill

log_folder = 'C:/_user/_other/diplo/code/navigation_neat/logs/go_to_target/1524777070'
gen_count = 300
pop_size = 100

fits = np.zeros((gen_count, pop_size))
collis = np.zeros((gen_count, pop_size))
dists = np.zeros((gen_count, pop_size))

for gen in range(0, gen_count):
    with open('{}/out-data-gen-{}'.format(log_folder, gen), 'rb') as f:
        (fits[gen, :], dists[gen, :], collis[gen, :]) = dill.load(f)

best_fits = np.amax(fits, axis=1) * -1  # need to change sign
best_fits_idx = np.argmax(fits, axis=1)

best_dists = np.zeros((gen_count, 1))
best_collis = np.zeros((gen_count, 1))
for gen in range(gen_count):
    best_dists[gen, 0] = dists[gen, best_fits_idx[gen]]
    best_collis[gen, 0] = collis[gen, best_fits_idx[gen]]

fig1 = plt.figure()
plt.plot(best_fits, 'go', markersize=3)
plt.plot(best_dists, 'bo', markersize=3)
plt.plot(best_collis, 'ro', markersize=3)

print('best collision count: {}'.format(np.min(best_collis)))
print('best distance position: {}'.format(np.min(best_dists)))
print('best fitness collision count: {}'.format(best_collis[299, 0]))
print('best fitness distance: {}'.format(best_dists[299, 0]))

plt.show(block=True)
