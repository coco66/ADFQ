import adfq_fun
import numpy as np

def iter_search_test():
    for _ in range(10): # Test on 10 different random examples.
        anum = 10
        noise = 0.0
        target_means = 100 * (np.random.random((anum,)) - 0.5)
        target_vars = 100 * np.random.random((anum,))
        c_mean = 100 * np.random.random()
        c_var = 100 * np.random.random()

        bar_vars = 1./(1./c_var + 1./(target_vars+noise))
        bar_means = bar_vars*(c_mean/c_var + target_means/(target_vars+noise))
        sorted_idx = np.flip(np.argsort(target_means), axis=-1)

        outputs = adfq_fun.posterior_adfq_helper(target_means, target_vars,
                            c_mean, c_var, discount=0.99, sorted_idx=sorted_idx)

        for b in range(anum):
            mu_star = outputs[b][0]
            b_primes = [c for c in sorted_idx if c!=b] # anum-1
            lhs = (mu_star - bar_means[b])/bar_vars[b]
            rhs = np.sum([(target_means[b_prime] - mu_star)/target_vars[b_prime] \
                    for b_prime in b_primes if target_means[b_prime] > mu_star])
            if np.abs(lhs - rhs) > 1e-2:
                print(i, rhs, lhs)
                import pdb; pdb.set_trace()

    print("PASS Mean*_b")

def batch_test():
    # Test example
    n_means = np.array(
        [[0.8811533, 0.9458812, 0.78800523, 0.7855228, 0.8548022,
        1.0071998, 0.8347546, 0.8921166, 0.9238528, 0.92809606,
        0.9720104, 0.8862572 ],
       [0.818557 , 0.7337841, 1.1270436, 1.239617 , 1.5188323,
        1.0134017, 1.0220736, 1.0529686, 1.1472604, 1.0060027,
        0.79905474, 1.0601841 ],
        [8.18557 , 0.7337841, 1.1270436, -1.239617 , 1.5188323,
         1.0134017, -10.220736, 1.0529686, -11.472604, 1.0060027,
         79.905474, 1.0601841 ]], dtype=np.float32)
    n_vars = np.array(
        [[56.72953, 50.670547, 57.65268, 57.328403, 58.38417, 57.436073,
        56.893364, 54.29608, 57.82789, 51.035233, 50.11123, 56.001694],
       [62.661198, 69.064156, 48.007538, 44.541973, 28.289375, 49.62694,
        56.438583, 51.90718, 51.234684, 46.22086, 53.456276, 41.784817],
        [0.626611, 69.064156, 48.007538, 0.44541973, 0.282893, 49.62694,
         56.438583, 51.90718, 0.051234, 46.22086, 0.053456, 41.784817]],
      dtype=np.float32)
    c_mean = np.array([0.9727963, 0.78518265, 6.2386910],dtype=np.float32)
    c_var = np.array([50.026917, 69.16801, 10.0012345], dtype=np.float64)
    reward = np.array([-0.0119087, -0., 1.0], dtype=np.float64)
    discount = 0.99
    terminal = [0, 1, 0]

    out_batch = adfq_fun.posterior_adfq(n_means, n_vars, c_mean, c_var,
                reward, discount, terminal=terminal, batch=True)
    for i in range(len(n_means)):
        out = adfq_fun.posterior_adfq(n_means[i], n_vars[i], c_mean[i], c_var[i],
                    reward[i], discount, terminal=terminal[i])
        if np.abs(out[0] - out_batch[0][i]) > 1e-5:
            print("MISMATCH Mean for ENTRY.%d:"%i, out[0], out_batch[0][i])
        else:
            print("PASS Mean for ENTRY.%d"%i)

        if np.abs(out[1] - out_batch[1][i]) > 1e-5:
            print("MISMATCH Variance for ENTRY.%d:"%i, out[1], out_batch[1][i])
        else:
            print("PASS Variance for ENTRY.%d"%i)

if __name__ == "__main__":
    batch_test()
    iter_search_test()
