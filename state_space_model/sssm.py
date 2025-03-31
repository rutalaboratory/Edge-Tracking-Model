import copy
import numpy as np
from scipy.special import logsumexp
from ssm.util import ssm_pbar, replicate, collapse
from ssm.messages import gaussian_logpdf, viterbi
from ssm.regression import fit_negative_binomial_integer_r
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from data_preprocess import load_preprocess_data
plt.rcParams.update({'font.size': 7,
                     'font.family': 'arial',
                     'pdf.fonttype': 42,
                     'ps.fonttype': 42,
                     'savefig.dpi': 480})


class SSSM(object):
    def __init__(self, seed=0, ndata=1, m0_share=False, m0_type='11', mt_type='22', vt_type='11',
                 _radius_bkw=1, _radius_fwd=1, _flter='none', _tau=1, _normalize=False,
                 _subwind=False, _dt=0.2):
        # number of states
        self.K = 2
        # observation and latent variable dimension
        self.D = 2
        # state-space paramters
        self.m0 = None
        self.sgm0 = None
        self.a_m = None
        self.b_m = None
        self.sigma_m = None
        self.a_v = None
        self.b_v = None
        self.sigma_v = None
        # input-determined switching process paramters
        self.p0 = None
        self.r_min, self.r_max = 1, 20
        self.r_011, self.r_100 = None, None
        self.p_011, self.p_100 = None, None
        self.map_d = None
        self.tmtrx_io = [None, None]
        self.semi_markov_nsample = 1  # set again in fit() for learning
        # random state
        self.rand_state = np.random.RandomState(seed)
        # learning kwargs
        self._radius_bkw = _radius_bkw
        self._radius_fwd = _radius_fwd
        self._flter = _flter
        self._tau = _tau
        self._normalize = _normalize
        self._subwind = _subwind
        self._subbias = np.array([[0, 0], [0, 1 * _dt]]) if _subwind else np.zeros((2, 2))
        self._dt = _dt
        # number of trials, number of m0
        self.ndata = ndata
        self.m0_share = m0_share
        self.m0_type = m0_type
        self.mt_type = mt_type
        self.vt_type = vt_type

    def initialize_params(self, bv_scale=1.0):
        # initial distribution of continuous latent variable
        self.m0 = (self.rand_state.rand(self.ndata, self.K, self.D) - 0.5)
        self.sgm0 = np.ones((self.ndata, self.K)) * 0.5
        for k in range(self.K):
            if self.m0_type[k] == '0':
                self.m0[:, k, :] = 0
                self.sgm0[:, k] = 0
        # state-space parameters
        self.a_m = np.ones(self.K) * 0.5
        self.b_m = np.ones(self.K) * 0.5
        self.sigma_m = np.ones(self.K) * 0.5
        self.a_v = np.ones(self.K) * 0.5
        self.b_v = np.ones(self.K) * 0.5 * bv_scale
        self.sigma_v = np.ones(self.K) * 0.5
        # input-determined switching process paramters
        self.p0 = np.ones(self.K) / self.K
        self.r_011, self.r_100 = 1, 1
        self.p_011 = self.rand_state.uniform(low=0.1, high=0.9)
        self.p_100 = self.rand_state.uniform(low=0.1, high=0.9)
        self.map_d = self.map_d_fn()
        for io in range(2):
            self.tmtrx_io[io] = self.tmtrx_io_fn(io)
        return

    def map_d_fn(self):
        return np.repeat(np.arange(self.K), [self.r_100, self.r_011])

    def tmtrx_io_fn(self, io):
        r_arr = [self.r_100, self.r_011]
        p_arr = [self.p_100, 1.] if io else [1., self.p_011]
        tmtrx = np.zeros((sum(r_arr), sum(r_arr)))
        ends = np.cumsum(r_arr)
        starts = np.concatenate(([0], ends[:-1]))
        for k1 in range(self.K):
            for k2 in range(self.K):
                block = tmtrx[starts[k1]:ends[k1], starts[k2]:ends[k2]]
                if k1 == k2:
                    if k1 != io:
                        for dr in range(r_arr[k1]):
                            block += (1 - p_arr[k1])**dr * p_arr[k1] * np.diag(np.ones(r_arr[k1]-dr), k=dr)
                    else:
                        block[:, 0] = 1.0
                else:
                    block[:, 0] = (1 - p_arr[k1]) ** np.arange(r_arr[k1], 0, -1) * 1
        assert np.allclose(tmtrx.sum(1), 1)
        assert (0 <= tmtrx).all() and (tmtrx <= 1.).all()
        return tmtrx

    def tmtrx_t_fn(self, inpt):
        T = len(inpt)
        tmtrx_t = np.empty((T - 1, len(self.map_d), len(self.map_d)))
        for t in range(1, T):
            tmtrx_t[t - 1] = self.tmtrx_io[int(inpt[t])]
        return tmtrx_t

    def most_likely_states(self, inpt, log_q_tk):
        tmtrx_t = self.tmtrx_t_fn(inpt)
        z_star = viterbi(replicate(self.p0, self.map_d), tmtrx_t, replicate(log_q_tk, self.map_d))
        return self.map_d[z_star]

    def direction_fn(self, obsv, t, k):
        _radius = self._radius_bkw + self._radius_fwd - 1
        if self._flter in ['exponential']:
            arr = obsv[max(0, t - _radius + 1): (t + 1)]
            arr = np.pad(arr, ((-min(0, t - _radius + 1), 0), (0, 0)), 'edge')
            weights = np.arange(t - _radius + 1, t + 1)
            weights = np.exp(-np.abs(weights - t) / self._tau)
            weights = weights / weights.sum()
            val = weights @ arr
        elif self._flter in ['average']:
            val = obsv[max(0, t - _radius + 1): (t + 1)].mean(0)
        elif self._flter == 'none':
            val = obsv[t]
        else:
            raise ValueError(f'{self._flter} not implemented.')
        if self._normalize:
            val = val / np.linalg.norm(val) * self._dt
        val = val - self._subbias[k]
        return val

    def cond_learn_fn(self, inpt, t, k):
        cond_stim = np.concatenate([np.zeros(self._radius_bkw), np.ones(self._radius_fwd)])
        cond_stim = (1 - cond_stim) if k == 1 else cond_stim
        cond_learn = np.array_equal(cond_stim, inpt[max(0, t - len(cond_stim) + 1): (t + 1)])
        return cond_learn

    def forward_continuous(self, obsv, inpt, idata):
        mus = np.zeros((2, len(obsv), 2))
        flags = np.zeros((2, len(obsv)))
        for t in range(len(obsv)):
            if t == 0:
                for k in range(2):
                    mus[k, 0, :] = self.m0[idata, k]
            else:
                for k in range(2):
                    cond_learn = self.cond_learn_fn(inpt, t, k)
                    if cond_learn:
                        flags[k, t] = 1
                    a_m_t = self.a_m[k] if cond_learn else 1.
                    b_m_t = self.b_m[k] if cond_learn else 0.
                    mus[k][t] = a_m_t * mus[k][t - 1] + b_m_t * self.direction_fn(obsv, t - 1, k)
        return mus, flags

    def kalman_filter(self, obsv, inpt, k, h_t, idata):
        m0 = self.m0[idata, k]
        sgm0 = self.sgm0[idata, k]
        a_m = self.a_m[k]
        b_m = self.b_m[k]
        sigma_m = self.sigma_m[k]
        a_v = self.a_v[k]
        b_v = self.b_v[k]
        sigma_v = self.sigma_v[k]
        T, D = obsv.shape
        predicted_mus = np.zeros((T, D))  # preds E[x_t | y_{1:t-1}]
        predicted_sigmas = np.zeros(T)  # preds Cov[x_t | y_{1:t-1}]
        filtered_mus = np.zeros((T, D))  # means E[x_t | y_{1:t}]
        filtered_sigmas = np.zeros(T)  # means Cov[x_t | y_{1:t}]
        # Initialize
        predicted_mus[0] = m0
        predicted_sigmas[0] = sgm0
        normalizer = 0
        for t in range(T):
            obsv_pre = obsv[t - 1] if t > 0 else obsv[0]

            # update normalizing constant
            v_err = obsv[t] - a_v * obsv_pre - b_v * predicted_mus[t]
            sgm = predicted_sigmas[t]
            normalizer += - h_t[t] * np.log(2 * np.pi * sigma_v)
            normalizer += np.log(sigma_v) - np.log(h_t[t] * b_v * sgm * b_v + sigma_v)
            normalizer -= v_err @ v_err * h_t[t] / 2 / (h_t[t] * b_v * sgm * b_v + sigma_v)

            # condition on this frame's observations
            m, sgm = predicted_mus[t], predicted_sigmas[t]
            kalman = (h_t[t] * sgm * b_v) / (h_t[t] * b_v * sgm * b_v + sigma_v)
            filtered_mus[t] = m + kalman * (obsv[t] - a_v * obsv_pre - b_v * m - b_v * self._subbias[k])
            filtered_sigmas[t] = (1 - kalman * b_v) * sgm
            if t == T - 1:
                break
            # predict
            cond_learn = self.cond_learn_fn(inpt, t + 1, k)
            a_m_t = a_m if cond_learn else 1.
            b_m_t = b_m if cond_learn else 0.
            sigma_m_t = sigma_m if cond_learn else 0.
            m, sgm = filtered_mus[t], filtered_sigmas[t]
            predicted_mus[t + 1] = a_m_t * m + b_m_t * self.direction_fn(obsv, t, k)
            predicted_sigmas[t + 1] = a_m_t * sgm * a_m_t + sigma_m_t
        return filtered_mus, filtered_sigmas, predicted_mus, predicted_sigmas, normalizer

    def kalman_smoother(self, obsv, inpt, k, h_t, idata):
        T, D = obsv.shape
        filtered_mus, filtered_sigmas, predicted_mus, predicted_sigmas, normalizer = self.kalman_filter(obsv, inpt, k, h_t, idata)
        smoothed_mus = np.zeros((T, D))
        smoothed_sigmas = np.zeros(T)
        EzTz0 = np.zeros(T)
        EzTz1 = np.zeros(T - 1)
        smoothed_mus[-1] = filtered_mus[-1]
        smoothed_sigmas[-1] = filtered_sigmas[-1]
        for t in range(T - 2, -1, -1):
            cond_learn = self.cond_learn_fn(inpt, t + 1, k)
            a_m_t = self.a_m[k] if cond_learn else 1.
            gt = (filtered_sigmas[t] * a_m_t / predicted_sigmas[t + 1]) if predicted_sigmas[t + 1] != 0 else 0
            smoothed_mus[t] = filtered_mus[t] + gt * (smoothed_mus[t + 1] - predicted_mus[t + 1])
            smoothed_sigmas[t] = filtered_sigmas[t] + gt * (smoothed_sigmas[t + 1] - predicted_sigmas[t + 1]) * gt
            EzTz1[t] = np.trace(
                gt * smoothed_sigmas[t + 1] * np.eye(D) + np.outer(smoothed_mus[t], smoothed_mus[t + 1]))
        for t in range(T):
            EzTz0[t] = np.trace(smoothed_sigmas[t] * np.eye(D) + np.outer(smoothed_mus[t], smoothed_mus[t]))
        return smoothed_mus, EzTz0, EzTz1, normalizer

    def compute_q_tk(self, obsv, Estats_c):
        q_tk = np.zeros((len(obsv), self.K))
        for k in range(self.K):
            smoothed_mus, EzTz0, _, _ = Estats_c[k]
            obsv_pre = np.concatenate([obsv[0:1], obsv[:-1]], axis=0)
            v_err = obsv - self.a_v[k] * obsv_pre - self.b_v[k] * self._subbias[k]
            log_qt = (-1 / 2) * np.einsum('td,td->t', v_err, v_err)
            log_qt += np.einsum('td,td->t', v_err, self.b_v[k] * smoothed_mus)
            log_qt += (-1 / 2) * np.square(self.b_v[k]) * EzTz0
            log_qt = log_qt / self.sigma_v[k] - np.log(2 * np.pi) - np.log(self.sigma_v[k])
            q_tk[:, k] = np.exp(log_qt)
        return q_tk

    def forward_backward(self, obsv, inpt, q_tk, discrete_state_type):
        T = len(obsv)
        tmtrx_t = self.tmtrx_t_fn(inpt)
        with np.errstate(divide="ignore"):
            log_tmtrx_t = np.log(tmtrx_t)
            log_p0 = replicate(np.log(self.p0), self.map_d)
            log_likes = replicate(np.log(q_tk), self.map_d)
        # forward
        alphas = np.zeros(log_likes.shape)
        alphas[0] = log_p0 + log_likes[0]
        for t in range(T - 1):
            m = np.max(alphas[t])
            with np.errstate(divide="ignore"):
                alphas[t + 1] = np.log(np.dot(np.exp(alphas[t] - m), tmtrx_t[t])) + m + log_likes[t + 1]
        normalizer = logsumexp(alphas[-1])
        # backward
        betas = np.zeros(log_likes.shape)
        betas[T - 1] = 0
        for t in range(T - 2, -1, -1):
            tmp = log_likes[t + 1] + betas[t + 1]
            m = np.max(tmp)
            betas[t] = np.log(np.dot(tmtrx_t[t], np.exp(tmp - m))) + m
        # expected states
        expected_states = alphas + betas
        expected_states -= logsumexp(expected_states, axis=1, keepdims=True)
        expected_states = np.exp(expected_states)
        expected_states = collapse(expected_states, self.map_d)
        # expected joints
        expected_joints = alphas[:-1, :, None] + betas[1:, None, :] + log_likes[1:, None, :] + log_tmtrx_t
        expected_joints -= expected_joints.max((1, 2))[:, None, None]
        expected_joints = np.exp(expected_joints)
        expected_joints /= expected_joints.sum((1, 2))[:, None, None]
        expected_joints = collapse(collapse(expected_joints, self.map_d, axis=2), self.map_d, axis=1)
        # posterior sample
        samples = self.backward_sample(tmtrx_t, alphas) if discrete_state_type == 'semi_markov' else None
        return expected_states, expected_joints, samples, normalizer

    def backward_sample(self, tmtrx_t, alphas):
        # posterior sample
        T, K_ = alphas.shape
        nsample = self.semi_markov_nsample
        u_arr = self.rand_state.rand(nsample, T)
        z_arr = - np.ones((nsample, T), dtype=int)
        lpzp1 = np.zeros((nsample, K_))
        for t in range(T - 1, -1, -1):
            lpz = lpzp1 + alphas[t]
            Z = logsumexp(lpz, axis=1)
            z_arr[:, t] = K_ - 1
            acc_pre = np.zeros(nsample)
            for k_ in range(K_):
                acc_cur = acc_pre + np.exp(lpz[:, k_] - Z)
                ismpls = np.argwhere((acc_pre <= u_arr[:, t]) & (u_arr[:, t] < acc_cur)).flatten()
                z_arr[ismpls, t] = k_
                acc_pre = acc_cur
            if t > 0:
                with np.errstate(divide="ignore"):
                    lpzp1 = np.take(np.log(tmtrx_t[(t - 1), :, :]), z_arr[:, t], axis=-1).T
        z_arr = self.map_d[z_arr.flatten()].reshape((nsample, T))
        return z_arr

    def compute_h_tk(self, Estats_d):
        h_tk, _, _, _ = Estats_d
        return h_tk

    def solve_linear(self, x_term, sq_term0, sq_term1, cnt, update_type):
        if update_type == '1':
            x_term_ = x_term[0] - x_term[1] - sq_term0[0, 1] + sq_term0[1, 1]
            sq_term0_ = sq_term0[0, 0] - sq_term0[0, 1] - sq_term0[1, 0] + sq_term0[1, 1]
            a = x_term_ / sq_term0_
            b = 1 - a
            ab = np.array([a, b])
        elif update_type == '2':
            ab = np.linalg.solve(sq_term0, x_term)
            a, b = ab[0], ab[1]
        else:
            raise ValueError(f'update type {update_type} not implemented.')
        sigma = (sq_term1 - 2 * x_term.T @ ab + ab.T @ sq_term0 @ ab) / cnt / 2
        return a, b, sigma

    def objective_continuous(self, obsvs, inpts, Estats_c_list):
        obj = [0 for _ in range(self.K)]
        for idata in range(self.ndata):
            obsv, inpt, Estats_c = obsvs[idata], inpts[idata], Estats_c_list[idata]
            for k in range(self.K):
                mus, EzTz0, EzTz1, ll = Estats_c[k]
                if self.m0_type[k] != '0':
                    log_m0 = EzTz0[0] - 2 * mus[0].T @ self.m0[idata, k] + self.m0[idata, k].T @ self.m0[idata, k]
                    log_m0 = - log_m0 / 2 / self.sgm0[idata, k] - np.log(2 * np.pi) - np.log(self.sgm0[idata, k])
                    obj[k] += log_m0
                for t in range(len(obsv) - 1):
                    cond_learn = self.cond_learn_fn(inpt, t + 1, k)
                    if cond_learn:
                        v_t = self.direction_fn(obsv, t, k)
                        log_mt = (EzTz0[t + 1] + np.square(self.a_m[k]) * EzTz0[t] +
                                  np.square(self.b_m[k]) * v_t.T @ v_t)
                        log_mt += (-2 * self.a_m[k] * EzTz1[t] - 2 * self.b_m[k] * mus[t + 1].T @ v_t +
                                   2 * self.a_m[k] * self.b_m[k] * mus[t].T @ v_t)
                        log_mt = - log_mt / 2 / self.sigma_m[k] - np.log(2 * np.pi) - np.log(self.sigma_m[k])
                        obj[k] += log_mt
        return np.array(obj)

    def mstep_continuous_state(self, obsvs, inpts, Estats_c_list):
        for k in range(self.K):
            ##########################################################
            if self.m0_share:
                m0_k, sgm0_k, cnt_0 = np.zeros(self.D), 0, 0
                for idata in range(self.ndata):
                    mus, EzTz0, _, _ = Estats_c_list[idata][k]
                    m0_k += mus[0]
                    sgm0_k += EzTz0[0]
                    cnt_0 += 1
                for idata in range(self.ndata):
                    self.m0[idata, k] = m0_k / cnt_0
                    self.sgm0[idata, k] = sgm0_k / cnt_0 / 2 - self.m0[idata, k].T @ self.m0[idata, k] / 2
            else:
                for idata in range(self.ndata):
                    mus, EzTz0, _, _ = Estats_c_list[idata][k]
                    self.m0[idata, k] = mus[0]
                    self.sgm0[idata, k] = (EzTz0[0] - mus[0].T @ mus[0]) / 2
            ##########################################################
            x_term, sq_term0, sq_term1, cnt = np.zeros(2), np.zeros((2, 2)), 0, 0
            for idata in range(self.ndata):
                obsv, inpt = obsvs[idata], inpts[idata]
                mus, EzTz0, EzTz1, _ = Estats_c_list[idata][k]
                for t in range(len(obsv) - 1):
                    cond_learn = self.cond_learn_fn(inpt, t + 1, k)
                    v_t = self.direction_fn(obsv, t, k)
                    if cond_learn:
                        x_term[0] += EzTz1[t]
                        x_term[1] += mus[t+1].T @ v_t
                        sq_term0[0, 0] += EzTz0[t]
                        sq_term0[0, 1] += mus[t].T @ v_t
                        sq_term0[1, 0] += mus[t].T @ v_t
                        sq_term0[1, 1] += v_t.T @ v_t
                        sq_term1 += EzTz0[t+1]
                        cnt += 1
            a, b, sigma = self.solve_linear(x_term, sq_term0, sq_term1, cnt, self.mt_type[k])
            self.a_m[k], self.b_m[k], self.sigma_m[k] = a, b, sigma
        return

    def simulation_from_xmems(self, zd, xmems, obsv, with_noise=True, sim_seed=0):
        rs = np.random.RandomState(seed=sim_seed)
        sim_steps = [obsv[0]]
        for it in range(1, len(obsv)):
            k = zd[it]
            cur = self.a_v[k] * sim_steps[-1] + self.b_v[k] * (xmems[k, it] + self._subbias[k])
            if with_noise:
                cur += np.sqrt(self.sigma_v[k]) * rs.randn(2)
            sim_steps.append(cur)
        return sim_steps

    def objective_output(self, obsvs, inpts, Estats_d_list, Estats_c_list):
        obj = [0 for _ in range(self.K)]
        for idata in range(self.ndata):
            obsv, inpt = obsvs[idata], inpts[idata]
            for k in range(self.K):
                h_t = Estats_d_list[idata][0][:, k]
                mus, EzTz0, _, _ = Estats_c_list[idata][k]
                obsv_pre = np.concatenate([obsv[0:1], obsv[:-1]], axis=0)
                v_err = obsv - self.a_v[k] * obsv_pre - self.b_v[k] * self._subbias[k]
                log_qt = (-1 / 2) * np.einsum('td,td->t', v_err, v_err)
                log_qt += np.einsum('td,td->t', v_err, self.b_v[k] * mus)
                log_qt += (-1 / 2) * np.square(self.b_v[k]) * EzTz0
                log_qt = log_qt / self.sigma_v[k] - np.log(2 * np.pi) - np.log(self.sigma_v[k])
                obj[k] += np.sum(h_t * log_qt)
        return np.array(obj)

    def mstep_output(self, obsvs, inpts, Estats_d_list, Estats_c_list):
        for k in range(self.K):
            x_term, sq_term0, sq_term1, cnt = np.zeros(2), np.zeros((2, 2)), 0, 0
            for idata in range(self.ndata):
                obsv, inpt = obsvs[idata], inpts[idata]
                h_t = Estats_d_list[idata][0][:, k]
                mus, EzTz0, _, _ = Estats_c_list[idata][k]
                mus_ = mus + self._subbias[k]
                EzTz0_ = EzTz0 + 2 * mus @ self._subbias[k] + self._subbias[k].T @ self._subbias[k]
                obsv_pre = np.concatenate([obsv[0:1], obsv[:-1]], axis=0)
                x_term[0] += np.einsum('t,td,td', h_t, obsv, obsv_pre)
                x_term[1] += np.einsum('t,td,td', h_t, obsv, mus_)
                sq_term0[0, 0] += np.einsum('t,td,td', h_t, obsv_pre, obsv_pre)
                sq_term0[0, 1] += np.einsum('t,td,td', h_t, obsv_pre, mus_)
                sq_term0[1, 0] += np.einsum('t,td,td', h_t, obsv_pre, mus_)
                sq_term0[1, 1] += np.einsum('t, t', h_t, EzTz0_)
                sq_term1 += np.einsum('t,td,td', h_t, obsv, obsv)
                cnt += np.sum(h_t)
            a, b, sigma = self.solve_linear(x_term, sq_term0, sq_term1, cnt, self.vt_type[k])
            self.a_v[k], self.b_v[k], self.sigma_v[k] = a, b, sigma
        return

    def objective_discrete_markov(self, inpts, Estats_d_list):
        obj = 0
        for idata in range(self.ndata):
            inpt, Estats_d = inpts[idata], Estats_d_list[idata]
            expected_states, expected_joints, _, _ = Estats_d
            with np.errstate(divide="ignore"):
                obj += np.nan_to_num(np.log(self.p0)) @ expected_states[0]
            for io in range(self.K):
                cur = np.sum(expected_joints[inpt[1:] == io], axis=0)
                obj += np.sum(np.log(self.tmtrx_io_fn(io)[1 - io]) * cur[1 - io])
        return obj

    def mstep_discrete_state_markov(self, inpts, Estats_d_list):
        p0 = 0
        tmtrx_io = [0, 0]
        for inpt, Estats_d in zip(inpts, Estats_d_list):
            expected_states, expected_joints, _, _ = Estats_d
            p0 += expected_states[0]
            for io in range(self.K):
                tmtrx_io[io] += np.sum(expected_joints[inpt[1:] == io], axis=0)
        self.p0 = p0 / p0.sum()
        for io in range(self.K):
            tmtrx = tmtrx_io[io]
            tmtrx = np.nan_to_num(tmtrx / tmtrx.sum(axis=-1, keepdims=True))
            tmtrx = np.where(tmtrx.sum(axis=-1, keepdims=True) == 0, 1.0 / self.K, tmtrx)
            self.tmtrx_io[io] = tmtrx
        self.r_100, self.r_011 = 1, 1
        self.p_100, self.p_011 = self.tmtrx_io[1][0, 0], self.tmtrx_io[0][1, 1]
        self.map_d = self.map_d_fn()
        return

    def mstep_discrete_state_semi_markov(self, inpts, Estats_d_list):
        dur_arr = {0: [], 1: []}
        r_arr = [None, None]
        p_arr = [None, None]
        p0 = 0
        for inpt, Estats_d in zip(inpts, Estats_d_list):
            expected_states, _, samples_d, _ = Estats_d
            p0 += expected_states[0]
            chng_idxs = [-1] + list(np.argwhere(np.diff(inpt)).flatten()) + [len(inpt) - 1]
            for i in range(len(chng_idxs) - 1):
                ist, ied = chng_idxs[i] + 1, chng_idxs[i + 1] + 1
                cur_io = int(inpt[ist])
                cnts = np.sum(samples_d[:, max(ist-1, 0):ied] == (1-cur_io), axis=1)
                cnts = cnts[cnts >= 1]
                dur_arr[1 - cur_io] += list(cnts)
        for io in range(2):
            if len(dur_arr[io]) == 0:
                r_arr[io], p_arr[io] = 1, 1e-2
            else:
                r_arr[io], p_arr[io] = fit_negative_binomial_integer_r(np.array(dur_arr[io]), self.r_min, self.r_max)
        self.p0 = p0 / p0.sum()
        self.r_100 = r_arr[0]
        self.r_011 = r_arr[1]
        self.p_100 = p_arr[0]
        self.p_011 = p_arr[1]
        self.map_d = self.map_d_fn()
        for io in range(2):
            self.tmtrx_io[io] = self.tmtrx_io_fn(io)
        return

    def fit(self, obsvs, inpts, init_h='inpt',
            init_scale_bv=1.0,
            phase0=100, phase1=1, tol=0.01,
            semi_markov_nsample=10, learning=True, history=True,
            verbose=2, **kwargs):
        if learning:
            self.initialize_params(bv_scale=init_scale_bv)
            self.semi_markov_nsample = semi_markov_nsample
        # initialize variational parameters
        ndata = len(obsvs)
        h_tk_list = [None for _ in range(ndata)]
        Estats_c_list = [None for _ in range(ndata)]
        q_tk_list = [None for _ in range(ndata)]
        Estats_d_list = [None for _ in range(ndata)]

        for idata in range(ndata):
            obsv, inpt = obsvs[idata], inpts[idata]
            h_tk = np.zeros((len(obsv), self.K))
            if init_h == 'linear':
                chng_idxs = [-1] + list(np.argwhere(np.diff(inpt)).flatten()) + [len(inpt) - 1]
                for i in range(len(chng_idxs) - 1):
                    ist, ied = chng_idxs[i] + 1, chng_idxs[i + 1] + 1
                    h_tk[ist:ied, 1] = np.linspace(1 - inpt[ist], inpt[ist], num=(ied - ist), endpoint=True)
                h_tk[:, 0] = 1 - h_tk[:, 1]
            else:
                h_tk[np.arange(len(inpt)), inpt] = 1
            h_tk_list[idata] = h_tk

        elbos = []
        params = []
        pbar = ssm_pbar(phase0 + phase1 + 1, verbose=verbose, description="ELBO: {:.1f}", prob=[0])
        for itr in pbar:
            discrete_state_type = 'markov' if itr < phase0 else 'semi_markov'
            elbo = 0
            for idata in range(ndata):
                obsv, inpt = obsvs[idata], inpts[idata]
                h_tk = h_tk_list[idata]
                Estats_c = []
                for k in range(self.K):
                    Estats_c.append(self.kalman_smoother(obsv, inpt, k, h_tk[:, k], idata))
                q_tk = self.compute_q_tk(obsv, Estats_c)
                Estats_d = self.forward_backward(obsv, inpt, q_tk, discrete_state_type)
                elbo += - np.sum(h_tk * np.log(q_tk)) + Estats_d[-1] + np.sum([cur[-1] for cur in Estats_c])
                Estats_c_list[idata] = Estats_c
                q_tk_list[idata] = q_tk
                Estats_d_list[idata] = Estats_d
            elbos.append(elbo)

            if verbose == 2:
                pbar.set_description("ELBO: {:.1f}".format(elbos[-1]))
            if itr == phase0 + phase1:
                break
            if learning:
                # obj = self.objective_continuous(obsvs, inpts, Estats_c_list).sum()
                self.mstep_continuous_state(obsvs, inpts, Estats_c_list)
                # d_elbo = self.objective_continuous(obsvs, inpts, Estats_c_list).sum() - obj
                # elbo = elbo + d_elbo
                # elbos.append(elbo)

                # obj = self.objective_output(obsvs, inpts, Estats_d_list, Estats_c_list).sum()
                self.mstep_output(obsvs, inpts, Estats_d_list, Estats_c_list)
                # d_elbo = self.objective_output(obsvs, inpts, Estats_d_list, Estats_c_list).sum() - obj
                # elbo = elbo + d_elbo
                # elbos.append(elbo)

                if discrete_state_type == 'markov':
                    self.mstep_discrete_state_markov(inpts, Estats_d_list)
                else:
                    self.mstep_discrete_state_semi_markov(inpts, Estats_d_list)

            for idata in range(ndata):
                h_tk_list[idata] = self.compute_h_tk(Estats_d_list[idata])

            if history:
                cur_dict = copy.deepcopy(self.__dict__)
                cur_dict.update({'discrete_state_type': (discrete_state_type == 'semi_markov')})
                params.append(cur_dict)
            if 0 < itr < phase0 and 0 < elbos[-1] - elbos[-2] < tol:
                phase0 = itr + 1
        return np.array(elbos), params, h_tk_list, Estats_c_list, q_tk_list, Estats_d_list


class DictClass:
    def __init__(self, dict_):
        for k in dict_:
            setattr(self, k, dict_[k])


def plot_colored_features(obsv, inpt, h_tk, zd, dt, fig_save_name, fig_save_fld='leave_return'):
    cumsum = np.cumsum(np.array([[0, 0]] + list(obsv)), axis=0).reshape(-1, 1, 2)  # [:-1]
    segments = np.concatenate([cumsum[:-1], cumsum[1:]], axis=1)
    speed = np.linalg.norm(obsv, axis=1) / dt
    vis_arrs = [inpt, h_tk[:, 1], zd, speed]
    vis_names = ['input', 'h_tk', 'zd', 'speed']
    fig, axes = plt.subplots(1, len(vis_arrs), figsize=(5 * len(vis_arrs), 5))
    for iax in range(len(vis_arrs)):
        vis_arr = vis_arrs[iax]
        lc = LineCollection(segments, cmap='viridis', norm=plt.Normalize(vis_arr.min(), vis_arr.max()))
        lc.set_array(vis_arr)
        lc.set_linewidth(0.5)
        line = axes[iax].add_collection(lc)
        fig.colorbar(line, ax=axes[iax])
        axes[iax].axis('equal')
        axes[iax].set_title(vis_names[iax])
    fig_file = f'./files/figures_ssm/{fig_save_fld}/{fig_save_name}_traj_fts.png'
    fig.savefig(fig_file, bbox_inches='tight')
    return


def plot_continuous_cmp(c_generative, c_posterior, c_gdth, dt, c_name, fig_save_name, fig_save_fld='leave_return'):
    fig, axes = plt.subplots(2, 1, figsize=(6, 6))
    T = c_generative.shape[1]
    for k in range(2):
        for ixy in range(2):
            axes[k].plot(np.arange(T) * dt, c_posterior[k][:, ixy],
                         color=f'C{ixy}', label='posterior' + '-' + ['x', 'y'][ixy])
            axes[k].plot(np.arange(T) * dt, c_generative[k][:, ixy],
                         color=f'C{ixy + 2}', label='generative' + '-' + ['x', 'y'][ixy])
            if c_gdth is not None:
                axes[k].plot(np.arange(T) * dt, c_gdth[k][:, ixy],
                             color='gray', alpha=0.5, linewidth=2)
            axes[k].legend()
    axes[0].set_ylabel(f'returning {c_name}')
    axes[1].set_ylabel(f'leaving {c_name}')
    axes[1].set_xlabel('time (second)')
    fig_file = f'./files/figures_ssm/{fig_save_fld}/{fig_save_name}_{c_name}.png'
    fig.savefig(fig_file, bbox_inches='tight', transparent=True)
    return


def plot_traj_base(cumsum, state, ax,
                   nT=None, init_xy=[0, 0],
                   colors=['C0', 'C1'], bds=[-0.5, 0.5, 1.5],
                   linewidth=0.25, alpha=1.0):
    cumsum = cumsum.reshape(-1, 1, 2)
    segments = np.concatenate([cumsum[:-1], cumsum[1:]], axis=1) + np.array(init_xy)[None, None, :]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bds, cmap.N)
    nT = len(state) if nT is None else nT
    lc = LineCollection(segments[:nT], cmap=cmap, norm=norm, alpha=alpha)
    lc.set_array(state[:nT])
    lc.set_linewidth(linewidth)
    _ = ax.add_collection(lc)
    return


def plot_traj_joint(cumsum, state, ax,
                    nT=None, init_xy=[0, 0],
                    colors=['C0', 'C1'],
                    linewidth=0.25, alpha=1.0):
    nT = len(state) if nT is None else nT
    state = state[:nT]
    cumsum = cumsum[:(nT + 1)]
    cumsum = cumsum + np.array(init_xy)[None, :]
    chng_idxs = [-1] + list(np.argwhere(np.diff(state)).flatten()) + [len(state)-1]
    for i in range(len(chng_idxs) - 1):
        ist, ied = chng_idxs[i]+1, chng_idxs[i+1]+1
        ax.plot(cumsum[ist:(ied+1), 0], cumsum[ist:(ied+1), 1],
                color=colors[state[ist]], alpha=alpha, linewidth=linewidth)
    return


def plot_sim_steps(steps_list, steps_names, state, fig_save_name, fig_save_fld='leave_return'):
    fig, axes = plt.subplots(1, len(steps_list), figsize=(5 * len(steps_list), 5))
    for ax, steps, steps_name in zip(axes, steps_list, steps_names):
        cumsum = np.cumsum(np.array([[0, 0]] + list(steps)), axis=0)
        plot_traj_base(cumsum, state, ax, colors=['C0', 'C1'], linewidth=0.5)
        ax.axis('equal')
        ax.set_title(steps_name)
    if fig_save_name is not None:
        fig_file = f'./files/figures_ssm/{fig_save_fld}/{fig_save_name}_sim_steps.png'
        fig.savefig(fig_file, bbox_inches='tight', transparent=True)
    return


def plot_vecs_base(state, vecs, cumsum, ax,
                   plot_type='goal', nT=None,
                   init_xy=[0, 0], alpha=0.8, vec_pos=[0.5, 0],
                   colors=['C0', 'C1'], plot_lens=[20, 20],
                   linewidth=0.3, head_width=4, zorder=100):
    nT = len(state) if nT is None else nT
    chng_idxs = [-1] + list(np.argwhere(np.diff(state[:nT])).flatten()) + [len(state[:nT]) - 1]
    for i in range(len(chng_idxs) - 1):
        it, ied = chng_idxs[i] + 1, chng_idxs[i + 1] + 1
        if plot_type == 'cross':
            if it - 2 < 0:
                continue
            k = state[it - 1]
            if not np.array_equal(np.array([k, k, 1-k, 1-k]), state[(it-2):(it+2)]):
                continue
            val = vecs[max(0, it - 2):(it + 1)].mean(0)
            val = val / np.linalg.norm(val)
            dx, dy = val[0], val[1]
            it_ = it - 2
        else:
            k = state[it]
            if vec_pos[k] == 0:
                it_ = it
            else:
                it_ = it + int((ied - it) * vec_pos[k])
            if len(vecs.shape) == 3:
                dx = vecs[k, it_, 0]
                dy = vecs[k, it_, 1]
            else:
                dx = vecs[it_, 0]
                dy = vecs[it_, 1]
        dx = dx * plot_lens[k]
        dy = dy * plot_lens[k]
        ax.arrow(cumsum[it_, 0] + init_xy[0], cumsum[it_, 1] + init_xy[1], dx, dy,
                 color=colors[k], linewidth=linewidth, alpha=alpha,
                 head_width=head_width, length_includes_head=True, zorder=zorder)
    return


def plot_edge(ax, deg, ymax, xmin, xmax, init_xy=[0, 0], ymin=0, type=2, dx=0,
              color=matplotlib.colormaps['Reds'](0.10), alpha=1):
    if deg == 90:
        ax.fill(np.array([-25, -25, 25, 25]) + init_xy[0],
                np.array([ymin, ymax, ymax, ymin]) + init_xy[1], color=color, alpha=alpha)
    if deg in [45, 135]:
        if type == 1:
            width1 = 25
            width2 = 2 * 25 * np.sqrt(2) - 25
        else:
            width1 = 25 * np.sqrt(2)
            width2 = 25 * np.sqrt(2)
        if deg == 45:
            ax.fill(np.array([ymin - width1, ymax - width1 + dx, ymax + width2 + dx, ymin + width2]) + init_xy[0],
                    np.array([ymin, ymax, ymax, ymin]) + init_xy[1], color=color, alpha=alpha)
        if deg == 135:
            ax.fill(np.array([-ymin - width1, -ymax - width1 + dx, -ymax + width2 + dx, -ymin + width2]) + init_xy[0],
                    np.array([ymin, ymax, ymax, ymin]) + init_xy[1], color=color, alpha=alpha)
    if deg == 0:
        ax.fill(np.array([xmin, xmin, xmax, xmax]) + init_xy[0],
                np.array([0, 50, 50, 0]) + init_xy[1], color=color, alpha=alpha)
    return


def load_train_data(filepath, config):
    if not hasattr(config, 'normalize'):
        config.normalize = False
    if not hasattr(config, 'phase'):
        config.phase = 'all'
    obsv, inpt, _, mode = load_preprocess_data(filepath, **config.preprocess_kwargs)
    if config.phase != 'all':
        idxs = np.sort(np.argwhere(mode == int(config.phase)).flatten())
        ist = idxs[0]
        ii_gap = np.argwhere(np.diff(idxs) > 1).flatten()
        ied = idxs[-1] + 1 if len(ii_gap) == 0 else idxs[ii_gap[0]] + 1
        obsv, inpt, mode = obsv[ist:ied], inpt[ist:ied], mode[ist:ied]
    ist = np.argwhere(inpt).flatten().min()
    obsv = obsv[ist:]
    obsv = (obsv / np.linalg.norm(obsv, axis=1, keepdims=True)) if config.normalize else obsv
    inpt = inpt.astype(int)[ist:]
    return obsv, inpt


if __name__ == '__main__':
    import os
    import pickle

    config_file = 'fly_c_dt0pt2_et45I1_sssm_initm01_rb3rf3_drctAvgNorm_h0linear_trN200N1_sd0'
    retrain = True
    verbose = 2
    if os.path.exists('./files/configs/' + config_file + '.pkl'):
        with open('./files/configs/' + config_file + '.pkl', 'rb') as f:
            config = pickle.load(f)
        config = DictClass(config)
        print('config file ' + config_file + ' loaded.', flush=True)
    else:
        raise FileNotFoundError(f'config file {config_file} does not exist.')

    obsvs = []
    inpts = []
    gdths = []
    if config.data_type == 'simulation':
        for filepath, trials in config.filepath_list:
            with open(filepath, 'rb') as f:
                res_list = pickle.load(f)
            for itrial in trials:
                res = res_list[itrial]
                obsv = res['steps']
                inpt = res['stims']
                obsvs.append(obsv)
                inpts.append(inpt.astype(int))
                gdths.append(res)
    else:
        for filepath in config.filepath_list:
            obsv, inpt = load_train_data(filepath, config)
            obsvs.append(obsv)
            inpts.append(inpt)
            gdths.append(None)
    ########################################################################################
    model_file = f'./files/models_ssm/{config.data_tr_name}_{config.model_name}.pkl'
    if os.path.isfile(model_file) and (not retrain):
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
            fit_ret = pickle.load(f)
    else:
        model = SSSM(**config.model_kwargs)
        fit_ret = model.fit(obsvs, inpts, **config.model_tr_kwargs, verbose=verbose)
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
            pickle.dump(fit_ret, f)
    ########################################################################################
    elbos, params, h_tk_list, Estats_c_list, q_tk_list, Estats_d_list = fit_ret
    print(np.round(elbos[-1], 3), np.round(max(elbos), 3))
    print('m0=', np.round(model.m0, 3))
    print('r_100=', np.round(model.r_100, 3))
    print('r_011=', np.round(model.r_011, 3))
    print('p_100=', np.round(model.p_100, 3))
    print('p_011=', np.round(model.p_011, 3))
    print('a_m=', np.round(model.a_m, 3))
    print('b_m=', np.round(model.b_m, 3))
    print('sigma_m=', np.round(model.sigma_m, 3))
    print('a_v=', np.round(model.a_v, 3))
    print('b_v=', np.round(model.b_v, 3))
    print('sigma_v=', np.round(model.sigma_v, 3))

    for idata in range(len(obsvs)):
        obsv, inpt, gdth = obsvs[idata], inpts[idata], gdths[idata]
        h_tk, Estats_c, q_tk, Estats_d = h_tk_list[idata], Estats_c_list[idata], q_tk_list[idata], Estats_d_list[idata]

        xmems_generative, flags = model.forward_continuous(obsv, inpt, idata)
        xmems_posterior = np.array([Estats_c[k][0] for k in range(2)])

        zd = model.most_likely_states(inpt, np.log(q_tk))
        zd_gdth = None if gdth is None else gdth['stats']

        fig_save_name = f'{config.data_tr_name}_idata{idata}_{config.model_name}'
        plot_colored_features(obsv, inpt, h_tk, zd, config.dt, fig_save_name)
        plot_continuous_cmp(xmems_generative, xmems_posterior, None, config.dt, 'mem', fig_save_name)
        sim_steps_mg = model.simulation_from_xmems(zd, xmems_generative, obsv, with_noise=False, sim_seed=0)
        sim_steps_mp = model.simulation_from_xmems(zd, xmems_posterior, obsv, with_noise=False, sim_seed=0)
        plot_sim_steps([obsv, sim_steps_mp, sim_steps_mg],
                       ['data', 'posterior', 'generative'], inpt, fig_save_name)
        # plot_example_traj(obsv, inpt, zd, xmems_posterior, fig_save_name)
        # colors_gdth = [matplotlib.colormaps['Blues'](0.8), matplotlib.colormaps['Oranges'](0.8)]
        # plot_example_traj_gdth(obsv, inpt, zd, xmems_posterior, gdth, fig_save_name, colors_gdth=colors_gdth)
        if idata >= 10:
            break