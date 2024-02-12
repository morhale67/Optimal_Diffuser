import matplotlib.pyplot as plt
import numpy as np
import os
from progress.bar import Bar
import pylops
from scipy.io import loadmat
from PIL import Image
from scipy.signal import fftconvolve
import itertools

niter_out = 20
niter_in = 10

s, mu, l = 4, 1, 50
lamda = [l, l]
num_patterns = 50
result_save_path = f'temp/TV/obj_1/{num_patterns}_real_{s}_BucketSize/'

os.makedirs(result_save_path, exist_ok=True)

JJ = np.arange(54, 63, s)
KK = np.arange(78, 90, s)
rec = np.zeros([s * 14 * len(JJ), s * 14 * len(KK)])

realz = np.random.permutation(np.arange(1190))[:num_patterns]

Dop = [
    pylops.FirstDerivative(
        (s * 14, s * 14), axis=0, edge=False, kind="forward", dtype=np.float64
    ),
    pylops.FirstDerivative(
        (s * 14, s * 14), axis=1, edge=False, kind="forward", dtype=np.float64
    ),
]

bar = Bar('reconstructiong...' + 's, mu, lamda = ' + str([s, mu, l]), max=len(JJ) * len(KK))
d1, d2 = -46, -36
shift = [d1, d2]
for ind_1, jj in enumerate(JJ):

    for ind_2, kk in enumerate(KK):
        jj = int(jj)
        kk = int(kk)
        t = np.sum(TEST[jj:jj + s, kk:kk + s, :], axis=(0, 1))

        r = np.zeros([14 * s, 14 * s, rael_num])
        for ind, nn in enumerate(range(rael_num)):
            delta = [test_pos[nn][0] - test_pos[test_num][0], test_pos[nn][1] - test_pos[test_num][1]]
            rr = get_ref([jj, kk], pos_1, delta, diffuser, s, shift);
            r[:, :, ind] = rr

        r = r[:, :, realz]
        r = r - np.repeat(r.mean(axis=2)[:, :, np.newaxis], num_patterns, axis=2)

        t = t[realz]

        r = np.reshape(r, [s * 14 * s * 14, num_patterns], order='F')
        r = np.transpose(r)
        r = pylops.MatrixMult(r)  # an object that represents a matrix multiplication operation using the MatrixMult
        # class and storing it in the variable r

        xinv = pylops.optimization.sparsity.splitbregman(
            r,
            t.ravel(),
            Dop,
            niter_outer=niter_out,
            niter_inner=niter_in,
            mu=mu,
            epsRL1s=lamda,
            tol=1e-4,
            tau=1,
            show=False,
            **dict(iter_lim=5, damp=1e-4)
        )[0]
        x_out = np.transpose(xinv.reshape(s * 14, s * 14))

        ind_x = ind_1 * 14 * s
        ind_y = ind_2 * 14 * s
        # if jj == JJ[0]:
        #     ind_x = 0
        # elif jj == JJ[1]:
        #     ind_x = 56
        # else:
        #     ind_x = 112

        # if kk == KK[0]:
        #     ind_y = 0
        # elif kk == KK[1]:
        #     ind_y = 56
        # else:
        #     ind_y = 112

        rec[ind_1 * s * 14:ind_1 * s * 14 + s * 14, ind_2 * s * 14:ind_2 * s * 14 + s * 14] = x_out

        np.save(result_save_path + 'rec_%d_%d_4X4' % (jj, kk) + '.npy', x_out, allow_pickle=True)

        bar.next()
bar.finish()
fig, ax = plt.subplots(1, 2, num=1)
fig.suptitle('s, mu, lamda = ' + str([s, mu, l]))
ax[0].imshow(rec)
ax[1].imshow(np.kron(TEST[JJ[0]:JJ[-1] + s, KK[0]:KK[-1] + s, :].mean(axis=2), np.ones([14, 14])))
plt.show()
# fig.savefig(r'/home/sharonlabgpu/Desktop/server/or/HighResGI_new/TV/recs/rec' + str(counter) +
#             '.png')
# plt.close(1)
# counter +=1
