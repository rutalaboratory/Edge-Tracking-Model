import os
import numpy as np


class Agent(object):
    def __init__(self, dt=None,
                 dt_fit=None, r_011=None, p_011=None, r_100=None, p_100=None,
                 _radius_bkw=1, _radius_fwd=1, _flter='average',
                 _tau=None, _radius=None, sub_wind=False,
                 a_m=None, b_m=None, sigma_m=None, init_m=None, downwind_thres=0,
                 resample_exit_m=False,
                 goal_speeds=None, a_v=None, sigma_v=None,
                 tau_a1=9.8, tau_r1=0.72, beta=0.01, stim_thres=0.25, **kwargs):
        self.dt = dt
        ####### parameters for odor filtering and binarizing. (alvarez-salvado et al 2018)
        self.exp_tau_a1 = np.exp(- dt / tau_a1)  # for odor adaptation factor
        self.exp_tau_r1 = np.exp(- dt / tau_r1)  # for odor response
        self.beta = beta  # for odor compression
        self.stim_thres = stim_thres  # threshold to binarize odor
        ####### paramters for delay in state transition. negative binomial distribution
        ############ dt_fit: parameters are learned from data with time step dt_fit
        ############ r_011, p_011: duration of leaving state after exit
        ############ r_100, p_100: duration of returning state after entry
        self.r_arr = [r_011, r_100]
        self.p_arr = [p_011, p_100]
        self.delay_coef = int(dt_fit / dt)
        ####### whether to subtract wind direction from exit direction for memory update
        self.sub_wind = sub_wind
        ####### parameters for computing travel direction for memory update
        self._radius_bkw = _radius_bkw * int(dt_fit / dt)
        self._radius_fwd = _radius_fwd * int(dt_fit / dt)
        self._flter = _flter
        self._tau = _tau * int(dt_fit / dt) if _tau is not None else None
        self._radius = _radius * int(dt_fit / dt) if _radius is not None else None
        ####### parameters for memory update: m(t+1) = a_m * m(t) + b_m * v(t) + sqrt(sigma_m) * randn(2)
        self.a_m = a_m
        self.b_m = [(1 - a_mk) for a_mk in a_m] if b_m is None else b_m
        self.sigma_m = sigma_m
        ####### parameters for memory initialization
        self.init_m = init_m
        self.downwind_thres = downwind_thres
        ####### resample exit memory from initialization distribution
        self.resample_exit_m = resample_exit_m
        ####### goal speeds [returning, leaving]
        self.goal_speeds = goal_speeds
        ####### displacement update: step(t+1) = a_v * step(t) + (1 - a_v) * goal(t+1) + sqrt(sigma) * randn(2)
        ####### convert a_v, sigma_v from time step dt_fit to time step dt_sim
        a_v, sigma_v = a_v.copy(), sigma_v.copy()
        for io in range(2):
            a_v[io], _, sigma_v[io] = param_convert(a_v[io], 1 - a_v[io], sigma_v[io], dt_fit, dt)
        self.a_v = a_v
        self.sigma_v = sigma_v
        #######
        self.rs = None  # random state
        self.modes = None  # phase of the trial
        self.winds = None  # wind direction
        self.odors = None  # odor concentration
        self.oadpt = None  # odor adaptation factor (equation 8 in alvarez-salvado et al 2018)
        self.ocmpr = None  # odor compression (equation 9 in ...)
        self.orspn = None  # odor response (equation 10 in ...)
        self.stims = None  # stimuli (binarized odor)
        self.ichng = None  # the time step when stimuli changes
        self.delay = None  # delay of state transition
        self.stats = None  # states (leaving/returning)
        self.xmems = None  # crossing (entry and exit) memories.
        self.goals = None  # goal direction
        self.steps = None  # displacement
        self.xylocs = None  # xy location

    def reset(self, nT, rs):
        self.rs = rs
        self.modes = np.zeros((nT,)).astype(int)
        self.winds = np.zeros((nT, 2))
        self.odors = np.zeros((nT,))
        self.oadpt = np.zeros((nT,))
        self.ocmpr = np.zeros((nT,))
        self.orspn = np.zeros((nT,))
        self.stims = np.zeros((nT,)).astype(int)
        self.ichng = -1
        self.delay = None
        self.stats = np.zeros((nT,), dtype=int)
        self.xmems = np.zeros((2, nT, 2))
        self.goals = np.zeros((nT, 2))
        self.steps = np.zeros((nT, 2))
        self.xylocs = np.zeros((nT + 1, 2))
        return

    def simulate(self, env, rs_, T, odor_filter=False):
        env.reset()
        nT = int(T / self.dt)
        self.reset(nT, np.random.RandomState(rs_))
        self.xylocs[0] = env.init_loc(self.rs)
        for it in range(nT):
            self.odors[it], self.winds[it], self.modes[it] = env.sense(xyloc=self.xylocs[it], it=it, odors=self.odors,
                                                                       steps=self.steps, ichng=self.ichng,
                                                                       xylocs=self.xylocs)  # kwargs
            self.odor_response(it, odor_filter=odor_filter)
            self.state(it)
            self.memory(it)
            self.goal(it)
            self.step(it)
        return

    def odor_response(self, it, odor_filter=False):
        if odor_filter:
            self.oadpt[it] = self.exp_tau_a1 * self.oadpt[it - 1] + (1 - self.exp_tau_a1) * self.odors[it]
            self.ocmpr[it] = self.odors[it] / (self.odors[it] + self.beta + self.oadpt[it])
            self.orspn[it] = self.exp_tau_r1 * self.orspn[it - 1] + (1 - self.exp_tau_r1) * self.ocmpr[it]
            self.stims[it] = int(self.orspn[it] > self.stim_thres)
        else:
            self.stims[it] = self.odors[it]

        if self.stims[it] != self.stims[it - 1]:
            self.ichng = it - 1
            delay = self.rs.negative_binomial(self.r_arr[self.stims[it]], 1 - self.p_arr[self.stims[it]]) + 1
            self.delay = delay * self.delay_coef
        return

    def state(self, it):
        if it == 0:
            self.stats[it] = self.stims[it]
        elif (self.stats[it - 1] != self.stims[it]) and (it == self.ichng + self.delay):
            self.stats[it] = 1 - self.stats[it - 1]
        else:
            self.stats[it] = self.stats[it - 1]
        return

    def travel_direction(self, it):
        _radius = (self._radius_bkw + self._radius_fwd - 1) if self._radius is None else self._radius
        if self._flter in ['exponential']:
            arr = self.steps[max(0, it - _radius + 1): (it + 1)]
            arr = np.pad(arr, ((-min(0, it - _radius + 1), 0), (0, 0)), 'edge')
            weights = np.arange(it - _radius + 1, it + 1)
            weights = np.exp(-np.abs(weights - it) / self._tau)
            weights = weights / weights.sum()
            val = weights @ arr
        elif self._flter in ['average']:
            val = self.steps[max(0, it - _radius + 1): (it + 1)].mean(0)
        elif self._flter == 'none':
            val = self.steps[it]
        else:
            raise ValueError(f'{self._flter} not implemented.')
        val = val / np.linalg.norm(val)  # travel direction
        return val

    def init_memory(self, it, k):
        ####### xmems[0]: memory of entry, initialized as the null vector
        ####### xmems[1]: memory of exit, initialized to be upwind with a random cross wind component
        wind_ortho = np.array([self.winds[it, 1], - self.winds[it, 0]])  # cross wind direction
        vec = (self.rs.uniform(*self.init_m[k][0]) * wind_ortho +
               self.rs.uniform(*self.init_m[k][1]) * self.winds[it])
        if k == 1 and self.sub_wind:  # subtract wind direction from exit memory
            vec -= self.winds[it]
        return vec

    def memory(self, it):
        if it == 0:
            for k in range(2):
                self.xmems[k, it] = self.init_memory(it, k)
            return
        for k in range(2):  # k=0: entry; k=1: exit
            cond_stim = np.concatenate([np.zeros(self._radius_bkw), np.ones(self._radius_fwd)])
            cond_stim = (1 - cond_stim) if k == 1 else cond_stim
            cond_learn = np.array_equal(cond_stim, self.stims[max(0, it - len(cond_stim) + 1): (it + 1)])
            if cond_learn:  # update memory
                direction = self.travel_direction(it - 1)
                if k == 1 and (direction @ self.winds[it] < self.downwind_thres):  # ignore downwind exit
                    self.xmems[k][it] = self.xmems[k][it - 1]
                    continue
                if k == 1 and self.resample_exit_m:
                    self.xmems[k][it] = self.init_memory(it, k)
                    continue
                if k == 1 and self.sub_wind:  # subtract wind direction from exit direction
                    direction = direction - self.winds[it]
                self.xmems[k][it] = self.a_m[k] * self.xmems[k][it - 1] + self.b_m[k] * direction
                self.xmems[k][it] += np.sqrt(self.sigma_m[k]) * self.rs.randn(2)
            else:
                self.xmems[k][it] = self.xmems[k][it - 1]
        return

    def goal(self, it):
        self.goals[it] = self.xmems[self.stats[it]][it]  # returning: memory of entry; leaving: memory of exit;
        if self.stats[it] == 1 and self.sub_wind:
            self.goals[it] += self.winds[it]
        return

    def step(self, it):
        a = self.a_v[self.stats[it]]
        self.steps[it] = (a * self.steps[it - 1]
                          + (1 - a) * self.goals[it] * self.goal_speeds[self.stats[it]] * self.dt
                          + np.sqrt(self.sigma_v[self.stats[it]]) * self.rs.randn(2))
        self.xylocs[it + 1] = self.xylocs[it] + self.steps[it]
        return


def param_convert(a, b, sigma, dt_fit, dt_sim):
    ####### for displacement update equation: step(t+1) = a * step(t) + b * m(t+1) + sqrt(sigma) * randn(2)
    ####### convert parameters under time step dt_fit to time step dt_sim
    if dt_fit == dt_sim:
        return a, b, sigma
    else:
        a_ = np.power(a, dt_sim / dt_fit)
        b_ = b / (1 - a) * (1 - a_)
        sigma = sigma / np.square(dt_fit)  # velocity = step_fit / dt_fit
        sigma_ = sigma / (1 - np.square(a)) * (1 - np.square(a_))
        sigma_ = sigma_ * np.square(dt_sim)  # step_sim = velocity * dt_sim
        return a_, b_, sigma_


def param_from_sssm(model, single_trial=False, downwind_thres=0.5):
    param_dict = {'dt_fit': model._dt,
                  'r_011': model.r_011, 'p_011': model.p_011,
                  'r_100': model.r_100, 'p_100': model.p_100,
                  'a_m': model.a_m, 'sigma_m': np.zeros(2),
                  'goal_speeds': model.b_m / (1 - model.a_m),
                  'a_v': model.a_v, 'sigma_v': model.sigma_v,
                  '_flter': model._flter, '_tau': model._tau,
                  '_radius_bkw': model._radius_bkw, '_radius_fwd': model._radius_fwd,
                  'downwind_thres': downwind_thres}
    if single_trial:
        init_m = model.m0[0] / (model._dt * model.b_m / (1 - model.a_m))[:, None]
        param_dict['init_m'] = np.repeat(init_m[..., None], 2, axis=-1)
    else:
        init_m = np.zeros((2, 2, 2))
        for k in [1]:
            arr = model.m0[:, k, :] / (model.b_m[k] / (1 - model.a_m[k])) / model._dt
            xwind_abs_median = np.median(np.abs(arr[:, 0]))
            init_m[k, 0, :] = [-2 * xwind_abs_median, 2 * xwind_abs_median]
            init_m[k, 1, :] = np.median(arr[:, 1])
        param_dict['init_m'] = init_m
    return param_dict


def param_from_fixedbias(model):
    param_dict = {'dt_fit': model._dt,
                  'r_011': model.r_011, 'p_011': model.p_011,
                  'r_100': model.r_100, 'p_100': model.p_100,
                  'a_m': np.ones(2), 'sigma_m': np.zeros(2), 'goal_speeds': np.ones(2),
                  'a_v': model.a_v, 'sigma_v': model.sigma_v}
    init_m = model.b_v / (1 - model.a_v[:, None]) / model._dt
    param_dict['init_m'] = np.repeat(init_m[..., None], 2, axis=-1)
    return param_dict


if __name__ == '__main__':
    from simulation_envs import *
    import matplotlib.pyplot as plt
    import pickle
    from sssm import plot_traj_base, plot_edge
    plt.rcParams.update({'font.size': 6,
                         'font.family': 'arial',
                         'pdf.fonttype': 42,
                         'ps.fonttype': 42,
                         'savefig.dpi': 480})
    ################## param_from_sssm ##################
    name1, fld_str, fld_select = 'All', '0123', '0123'
    name2, init_str = 'N', '_initm01'
    save_file = (f'./files/models_ssm/fly_c_dt0pt2_fld{fld_str}_'
                 f'sssm{init_str}_rb3rf3_drctAvgNorm_h0linear_trN200N1_sd0.pkl')
    with open(save_file, 'rb') as f:
        ifldlist_to_sssm = pickle.load(f)
    model = ifldlist_to_sssm[fld_select]
    param_str = f'Fit{name1}{name2}0'
    param_dict = param_from_sssm(model, downwind_thres=0.5)
    print(param_str)
    print(param_dict)
    ######################## setting time step and environment ##############
    dt = 0.2
    odor_filter = False
    # env, T = EnvEdge(deg=45, half_width=25*np.sqrt(2), ymin=-np.inf), 1000
    env, T = EnvEdge(deg=90, half_width=25, ymin=-np.inf), 1000
    # env, T = EnvEdge(deg=0, half_width=25, ymin=-np.inf), 1000
    # env, T = EnvJump(), 600
    # env, T = EnvReplay(deg=0, T1=600, T2=610, dt=dt), 1210
    # env, T = EnvZigzag(deg1=45, deg2=135, deg3=135, y1=200, nx=[0, 3, 11, None][0],
    #                    dt=dt, dur_thres=2.0, dist_thres=4.0), 2000
    # env, T = EnvZigzag(deg1=45, deg2=45, deg3=135, y1=200, nx=[0, 3, 11, None][2],
    #                    dt=dt, dur_thres=2.0, dist_thres=4.0), 2000
    ######################## agent ########################
    agent = Agent(dt=dt, **param_dict)
    ######################## run simulation  ########################
    rerun = True
    save_file = f'./files/simulations/{param_str}_{env.env_name}.pkl'
    if os.path.isfile(save_file) and (not rerun):
        with open(save_file, 'rb') as f:
            res_list = pickle.load(f)
    else:
        res_list = {}
        for rs_ in range(5):  # random seed
            agent.simulate(env, rs_, T, odor_filter=odor_filter)
            res_list[rs_] = {'steps': agent.steps, 'stims': agent.stims, 'modes': agent.modes,
                             'stats': agent.stats, 'xmems': agent.xmems, 'goals': agent.goals,
                             'xyloc0': agent.xylocs[0]}
        # with open(save_file, 'wb') as f:
        #     pickle.dump(res_list, f)
    ####################### plot simulated trajectories #######################
    for rs_, res in res_list.items():
        figsize = (5, 5)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        stims = res['stims']
        nT = len(res['steps'])
        cumsum = np.cumsum(np.array([res['xyloc0']] + list(res['steps'][:nT])), axis=0)
        plot_traj_base(cumsum, res['stims'][:nT], ax,
                       nT=None, init_xy=[0, 0], colors=['midnightblue', 'r'], linewidth=0.5)
        ax.axis('equal')
        ax.set_title(rs_)
        fig.show()
        # fig_file = f'./files/figures_ssm/leave_return/sim_{param_str}_{env.env_name}_example{rs_}.pdf'
        # fig.savefig(fig_file, bbox_inches='tight', transparent=True)
