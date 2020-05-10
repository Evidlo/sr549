# %% monte

def monte_exp(*, dbsnr, repetition):
    # add noise to frames
    frames_noisy = add_noise(frames, dbsnr=dbsnr)

    est_drift = registration(frames_noisy) * f.factor

    rt = []
    for f1, f2 in zip(frames_noisy[:-1], frames_noisy[1:]):
        rt.append(tuple(register_translation(f1, f2, upsample_factor=40)[0]))

    rt_drift = np.array(rt).mean(axis=0)

    return {
        'true_drift': f.true_drift,
        'est_drift': tuple(est_drift),
        'est_x': est_drift[0],
        'est_y': est_drift[1],
        'err_m': np.linalg.norm(est_drift - f.true_drift),
        'err_rt': np.linalg.norm(rt_drift - f.true_drift)
    }

from mas.misc import combination_experiment

result = combination_experiment(
    monte_exp,
    dbsnr=np.linspace(-30, -10, 10),
    repetition=np.arange(30),
)

# %% plot

import seaborn as sns
sns.set()


# sns.scatterplot()


# plt.scatter(result.est_x, result.est_y)

result_long =  result.melt(
    id_vars=['dbsnr'],
    value_vars=['err_m', 'err_rt'],
    value_name='err',
    var_name='method'
)



ax = sns.lineplot(x='dbsnr', y='err', hue='method', data=result_long)
ax.set(ylabel='Error (px)', xlabel='SNR (dB)')
