import glob
import os

import numpy as np
# from plotting.plot import plot
from matplotlib import pyplot as plt

def get_latest_run_id(save_dir: str) -> int:
    max_run_id = 0
    for path in glob.glob(os.path.join(save_dir, 'run_[0-9]*')):
        filename = os.path.basename(path)
        ext = filename.split('_')[-1]
        if ext.isdigit() and int(ext) > max_run_id:
            max_run_id = int(ext)
    return max_run_id

def get_line_styles(name):

    linewidth = 1
    linestyle = '-'

    if name == 'no aug':
        linewidth = 3
        linestyle = '-'
    if name == 'no aug 64':
        linewidth = 3
        linestyle = '--'

    style_dict = {
        'linestyle': linestyle,
        'linewidth': linewidth,
        # 'color': color,
    }

    return style_dict


def load_data(path, name, success_threshold=None):
    with np.load(path) as data:
        # for key in data:
        #     print(key)

        try:
            t = data['t']
        except:
            t = data['timesteps']

        if name == 'se_normalized':
            r = np.clip(data['diff_kl_mle_target']/np.abs(data['ref_kl_mle_target']), -10000, 100000)
        elif name == 'diff_kl_mle_target':
            r = np.clip(data['diff_kl_mle_target'], -10000,10000)
        else:
            r = data[name]
        if success_threshold is not None:
            r = r > success_threshold

        if len(r.shape) > 1:
            avg = np.average(r, axis=1)
        else:
            avg = r

    return t, t, avg


def plot(save_dict, name, m=100000, success_threshold=None, return_cutoff=-np.inf):
    i = 0

    # palette = seaborn.color_palette()
    print(os.getcwd())

    for agent, info in save_dict.items():
        paths = info['paths']
        x_scale = info['x_scale']
        max_t = info['max_t']
        avgs = []
        for path in paths:
            u, t, avg = load_data(path, name=name, success_threshold=success_threshold)
            if avg is not None:
                if max_t:
                    cutoff = np.where(t <= max_t/x_scale)[0]
                    avg = avg[cutoff]
                    t = t[cutoff]

                elif m:
                    avg = avg[:m]
                avgs.append(avg)
                t_good = t

        if len(avgs) == 0:
            continue
        elif len(avgs) == 1:
            avg_of_avgs = avg
            q05 = np.zeros_like(avg)
            q95 = np.zeros_like(avg)

        else:

            min_l = np.inf
            for a in avgs:
                l = len(a)
                if l < min_l:
                    min_l = l

            if min_l < np.inf:
                for i in range(len(avgs)):
                    avgs[i] = avgs[i][:min_l]

            avg_of_avgs = np.mean(avgs, axis=0)

            # if avg_of_avgs.mean() > 0: continue
            # print(np.median(avg_of_avgs))
            # if np.median(avg_of_avgs) > 0: continue

            std = np.std(avgs, axis=0)
            N = len(avgs)
            ci = 1.96 * std / np.sqrt(N) * 1.96
            q05 = avg_of_avgs - ci
            q95 = avg_of_avgs + ci


            # if avg_of_avgs[-10:].mean() < 4900 or N < 10: continue

        style_kwargs = get_line_styles(agent)
        style_kwargs['linewidth'] = 2

        # style_kwargs['linewidth'] = 1.5

        style_kwargs['color'] = None
        # if 'PROPS' in agent:
        #     style_kwargs['linestyle'] = '-'
        #     style_kwargs['linewidth'] = 3
        #     # style_kwargs['color'] = 'k'
        #
        #
        # elif 'ppo_buffer' in agent or 'PPO-Buffer' in agent or 'b=' in agent or 'Buffer' in agent:
        #     style_kwargs['linestyle'] = '--'
        # elif 'ppo,' in agent or 'PPO,' in agent or 'PPO with' in agent or 'PPO' == agent:
        #     style_kwargs['linestyle'] = ':'
        # elif 'Priv' in agent:
        #     style_kwargs['linestyle'] = '-.'
        #
        # elif '0.0001' in agent:
        #     style_kwargs['linestyle'] = '--'

        # print(agent, N, avg_of_avgs[-1], q05[-1], q95[-1])

        try:
            times = info['times']
            x = times
        except:
            x = t_good * x_scale
            if t is None:
                x = np.arange(len(avg_of_avgs))
            if m:
                x = x[:m]
                avg_of_avgs = avg_of_avgs[:m]
                q05 = q05[:m]
                q95 = q95[:m]
        plt.plot(x, avg_of_avgs, label=agent, **style_kwargs)
        if style_kwargs['linestyle'] == 'None':
            plt.fill_between(x, q05, q95, alpha=0)
        else:
            plt.fill_between(x, q05, q95, alpha=0.2)
        # plt.fill_between(x, q05, q95, alpha=0.2, color=style_kwargs['color'])

        i += 1
    # return fig


# def get_paths(results_dir, key, n_trials=20):
#
#     path_dict = {}
#     path_dict[key] = []
#     for j in range(n_trials):
#         path_dict[key].append(f'./{results_dir}/run_{j+1}/evaluations.npz')
#     return path_dict

def get_paths(results_dir, key, x_scale=1, max_t=None, evaluations_name='evaluations'):
    # print(results_dir)
    path_dict = {}
    path_dict[key] = {
        'paths': [],
        'x_scale': 1,
        'max_t': max_t,
    }
    # for item in os.listdir(root):
    #     if os.path.isfile(os.path.join(root, item)):
    #         print
    #         item

    try:
        for subdir in os.listdir(results_dir):
            if 'run_' in subdir:
                path_dict[key]['paths'].append(f'{results_dir}/{subdir}/{evaluations_name}.npz')
                path_dict[key]['x_scale'] = x_scale
                path_dict[key]['max_t'] = max_t
    except Exception as e:
        # print(e)
        x = 0
    return path_dict



def get_plot_data(paths):
    n = 100000000

    avgs = []
    for path in paths:
        t, avg = load_data(path)
        if avg is not None:
            avgs.append(avg[:n])
    t = t[:n]
    if len(avgs) == 1:
        avg_of_avgs = avg
        q05 = np.zeros_like(avg)
        q95 = np.zeros_like(avg)

    else:
        avg_of_avgs = np.average(avgs, axis=0)
        std = np.std(avgs, axis=0)
        N = len(avgs)
        ci = 1.96 * std / np.sqrt(N)

    return t, avg_of_avgs, ci

def get_times(save_dict):
    print(os.getcwd())

    time_dict = {}
    for agent, info in save_dict.items():
        paths = info['paths']
        times = []
        for path in paths:
            with np.load(path, allow_pickle=True) as data:
                time = data['times']
                time = np.insert(time, 0, 0)
            if time is not None:
                times.append(time)

        if len(times) == 0:
            continue
        elif len(times) == 1:
            avg_time = time
        else:
            avg_time = np.mean(times, axis=0)
            std_time = np.std(times, axis=0)


        if not (avg_time is None):
            time_dict[agent] = (avg_time, std_time)

    return time_dict
