import os
import pickle
import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter1d
import pandas as pd
from scipy import interpolate


def open_log(logfile):
    df = pd.read_table(logfile, delimiter='[,]', engine='python')
    if 'timestamp -- motor_step_command' in df:
        new = df['timestamp -- motor_step_command'].str.split('--', n=1, expand=True)
        df['timestamp'] = new[0]
        df['motor_step_command'] = new[1]
        df.drop(columns=['timestamp -- motor_step_command'], inplace=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', format="%m/%d/%Y-%H:%M:%S.%f ")
        df['seconds'] = 3600 * df.timestamp.dt.hour + 60 * df.timestamp.dt.minute + \
                        df.timestamp.dt.second + 10 ** -6 * df.timestamp.dt.microsecond
    df['seconds'] -= df.iloc[0]['seconds']
    return df


def exclude_lost_tracking(data, thresh=10, verbose=False):
    jumps = np.sqrt(np.gradient(data.ft_posy) ** 2 + np.gradient(data.ft_posx) ** 2)
    resets, _ = signal.find_peaks(jumps, thresh)
    for ii, reset in enumerate(resets):
        if jumps[reset + 1] > thresh:
            resets[ii] += 1
    l_mod = np.concatenate(([0], resets.tolist(), [len(data)]))
    l_mod = l_mod.astype(int)
    list_of_dfs = [data.iloc[l_mod[n]:l_mod[n + 1]] for n in range(len(l_mod) - 1)]
    if len(list_of_dfs) > 1:
        data = max(list_of_dfs, key=len)
        data.reset_index()
        if verbose:
            print('---------- LOST TRACKING, SELECTION MADE', )
    return data


def reset_lost_tracking(data, thresh=10, verbose=False):
    x = data.ft_posx.to_numpy()
    y = data.ft_posy.to_numpy()
    dx = np.diff(x)
    dy = np.diff(y)
    ds = np.sqrt(dx ** 2 + dy ** 2)
    idxs = np.argwhere(ds > thresh).flatten()
    if len(idxs) > 0:
        if verbose:
            print(f'---------- LOST TRACKING, reset at idx {idxs}', flush=True)
        for idx in idxs:
            dx[idx] = 0
            dy[idx] = 0
            ds[idx] = 0
        data.ft_posx = x[0] + np.cumsum([0] + list(dx))
        data.ft_posy = y[0] + np.cumsum([0] + list(dy))
    return data


def consolidate_in_out(o_arr, t_arr, t_cutoff=0.5):
    o_arr = o_arr.copy()
    for io in range(2):
        chng_idxs = [-1] + list(np.argwhere(np.diff(o_arr)).flatten()) + [len(o_arr) - 1]
        for i in range(len(chng_idxs) - 1):
            ist, ied = chng_idxs[i] + 1, chng_idxs[i + 1] + 1
            if (t_arr[ied - 1] - t_arr[ist] < t_cutoff) and (o_arr[ist] == io):
                o_arr[ist:ied] = 1 - io
    return o_arr


def combine_phase_mode_sampled(data, t, t_):
    t_sampled = t_[:-1]
    modes = np.zeros(len(t_sampled))
    if 'mode' in data:
        flags = data['mode'].values
        flags = np.array([i == 'replay' for i in flags])
        if 'exp_phase' in data:
            flags = np.nan_to_num(data['exp_phase'].values) - 1
        flag_chngidxs = [-1] + list(np.argwhere(np.diff(flags)).flatten()) + [len(flags) - 1]
        for i in range(len(flag_chngidxs) - 1):
            ist, ied = flag_chngidxs[i] + 1, flag_chngidxs[i + 1]
            tst, ted = t[ist], t[ied]
            if ted - tst > 1:
                cond_t = (t_sampled >= tst) & (t_sampled <= ted)
                modes[cond_t] = flags[ist + 1]
    return modes


def interp_fixed_interval(t, arr, fixedintv):
    interp_func = interpolate.interp1d(t, arr, axis=0)
    t_ = np.arange(t[0], t[-1], fixedintv)
    arr_ = interp_func(t_)
    return t_, arr_


def load_preprocess_data(filepath, lost_reset=True, consolidate_io=True,
                         fixedintv=0.2, speed_thres=1,
                         flter='none', sigma=0.1, cutoff_freq=0.5, order=10,
                         alternative_return=None, return_heading=False):
    # load and preprocess
    data = open_log(filepath)
    data = reset_lost_tracking(data, thresh=10) if lost_reset else exclude_lost_tracking(data, thresh=10)
    if ('replay' in filepath) or ('disappearing' in filepath):
        data['instrip'] = np.where(data.mfc2_stpt > 0, True, False)
    if 'constant' in filepath.lower():
        data['instrip'] = np.where(data.mfc3_stpt > 0, True, False)
    if consolidate_io:
        data.instrip = consolidate_in_out(data.instrip.values, data.seconds.values)
    if data.iloc[0].seconds == data.iloc[1].seconds:
        data = data.drop(0)
    # sample
    t = data.seconds.to_numpy()
    x = data.ft_posx.to_numpy()
    y = data.ft_posy.to_numpy()
    h = data.ft_heading.to_numpy()
    h = np.unwrap(h)
    h = - h + np.pi / 2
    o = data.instrip.to_numpy()
    ##### filter steps
    if flter == 'none':
        t_, xyho_ = interp_fixed_interval(t, np.array([x, y, h, o]).T, fixedintv)
    else:
        dt0 = 0.05
        t_, xyho_ = interp_fixed_interval(t, np.array([x, y, h, o]).T, dt0)
        if flter == 'gaussian':
            xyho_[:, [0, 1]] = gaussian_filter1d(xyho_[:, [0, 1]], sigma=(sigma/dt0), axis=0, mode='nearest')
        elif flter == 'butterworth':
            nyquist_freq = 0.5 * (1 / dt0)
            cutoff = cutoff_freq / nyquist_freq
            b, a = butter(order, cutoff, btype='low')
            xyho_[:, [0, 1]] = filtfilt(b, a, xyho_[:, [0, 1]], axis=0)
        else:
            raise ValueError(f'{flter} not implemented.')
        t_, xyho_ = interp_fixed_interval(t_, xyho_, fixedintv)
    ##################
    txyho_sampled = np.concatenate([t_[:, None], xyho_], axis=1)
    steps = np.diff(txyho_sampled[:, [1, 2]], axis=0)
    stims = (txyho_sampled[:-1, -1] > 0.5)
    heads = txyho_sampled[:-1, -2]
    modes = combine_phase_mode_sampled(data, t, t_)
    dists = np.linalg.norm(steps, axis=1)
    dists_cond = (dists > speed_thres * fixedintv)
    steps = steps[dists_cond]
    stims = stims[dists_cond]
    heads = heads[dists_cond]
    modes = modes[dists_cond]
    if alternative_return is not None:
        arr = [data[k].to_numpy() for k in alternative_return]
        _, arr_ = interp_fixed_interval(t, np.array(arr).T, fixedintv)
        return steps, stims, heads, modes, arr_[:-1][dists_cond]
    if return_heading:
        steps[:, 0] = np.cos(heads)
        steps[:, 1] = np.sin(heads)
    return steps, stims, heads, modes


def edge_rmtrx(theta):
    # rotate counterclockwise by pi/2-theta
    rot_mtrx = np.array([[np.sin(theta), -np.cos(theta)],
                         [np.cos(theta), np.sin(theta)]])
    return rot_mtrx


def edge_range(theta):
    # return the edge direction in the range of [0, pi)
    theta = np.mod(theta + np.pi, 2 * np.pi) - np.pi  # [-pi, pi)
    theta = (theta + np.pi) if theta < 0 else theta
    return theta


def traj_properties(steps, stims, dt, modes=None, stats=None, goals=None, theta=None, calc=True):
    data_list = []
    chng_idxs = [-1] + list(np.argwhere(np.diff(stims)).flatten()) + [len(stims) - 1]
    modes = np.zeros(len(stims)) if modes is None else modes
    for i in range(len(chng_idxs) - 1):
        ist, ied = chng_idxs[i] + 1, chng_idxs[i + 1] + 1
        cur_dict = {'iseg': i, 'io': stims[ist], 'ist': ist, 'ied': ied, 'mode': modes[ied - 1],
                    'duration': (ied - ist) * dt, 'steps': steps[ist:ied]}
        cur_dists = np.linalg.norm(steps[ist:ied], axis=1)
        cur_dict['speed'] = cur_dists / dt
        cur_dict['dttl'] = np.sum(cur_dists)
        cur_dict['avgs'] = cur_dict['dttl'] / cur_dict['duration']
        if not calc:
            cur_dict['netd_x'] = np.sum(steps[ist:ied, 0])
            cur_dict['netd_y'] = np.sum(steps[ist:ied, 1])
        if stats is not None:
            cur_dict['stats'] = stats[ist:ied]
        if goals is not None:
            cur_dict['goals'] = goals[stims[ist]][ied - 1, :] if len(goals.shape) == 3 else goals[ied - 1, :]
        if calc:
            cur_cumsum = np.cumsum(np.array([[0, 0]] + list(steps[ist:ied])), axis=0)
            cur_dict['seg'] = cur_cumsum
            cur_theta = theta if theta is not None else np.arctan2(cur_cumsum[-1, 1], cur_cumsum[-1, 0])
            cur_theta = edge_range(cur_theta)
            cur_dict['theta'] = cur_theta
            rot_mtrx = edge_rmtrx(cur_theta)
            rot_cumsum = (rot_mtrx @ cur_cumsum[:, [0, 1]].T).T
            cur_dict['seg_rot'] = rot_cumsum
            for rot in [True, False]:
                for ixy in range(2):
                    d_str = ['ppd', 'prl'][ixy] if rot else ['x', 'y'][ixy]
                    seg = [cur_cumsum, rot_cumsum][int(rot)]
                    cur_dict[f'netd_{d_str}'] = seg[-1, ixy] - seg[0, ixy]
                    cur_dict[f'rng_{d_str}'] = max(seg[:, ixy]) - min(seg[:, ixy])
                    cur_dict[f'dttl_{d_str}'] = np.sum(np.abs(np.diff(seg[:, ixy])))
                    cur_dict[f'avgs_{d_str}'] = cur_dict[f'dttl_{d_str}'] / cur_dict['duration']
        data_list.append(cur_dict)
    data_df = pd.DataFrame(data=data_list)
    return data_df


def traj_properties_simple(steps, stims, sfx=''):
    exit_idxs = np.argwhere(np.diff(stims) == -1).flatten()
    entry_idxs = np.argwhere(np.diff(stims) == 1).flatten()
    traj_rng = np.linalg.norm(sum(steps[exit_idxs[0]:exit_idxs[-1]])) if len(exit_idxs) >= 2 else 0
    nsteps_on_edge = (exit_idxs[-1] - exit_idxs[0]) if len(exit_idxs) >= 2 else 0
    path_len = np.linalg.norm(steps, axis=1).sum()
    dx, dy = np.sum(steps[:, 0]), np.sum(steps[:, 1])
    cumsum = np.cumsum(np.array([np.zeros(2)] + list(steps)), axis=0)
    prop_dict = {f'num_exit{sfx}': len(exit_idxs), f'num_entry{sfx}': len(entry_idxs),
                 f'traj_rng{sfx}': traj_rng, f'path_len{sfx}': path_len, f'nsteps{sfx}': len(steps),
                 f'nsteps_on_edge{sfx}': nsteps_on_edge,
                 f'dx{sfx}': dx, f'dy{sfx}': dy, f'dy_min{sfx}': cumsum[:, 1].min()}
    return prop_dict