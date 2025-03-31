import numpy as np


class EnvBase(object):
    # base class for all environments
    def __init__(self):
        self.mode = 0   # phase of the trial
        self.wind = np.array([0., 1.])  # wind direction

    def reset(self):    # reset for every simulation
        return

    def init_loc(self, rs):     # initial location for the agent
        return np.array([0., 0.])


class EnvEdge(EnvBase):
    # an odor strip
    def __init__(self, deg=None, half_width=25, ymin=-np.inf):
        super(EnvEdge, self).__init__()
        self.deg = deg      # direction of the edge
        self.half_width = half_width    # half width of the odor strip
        if self.deg == 90:
            sense_fn = lambda xyloc: (np.abs(xyloc[0]) <= self.half_width) & (xyloc[1] >= ymin)
        elif self.deg == 45:
            sense_fn = lambda xyloc: (np.abs(xyloc[0] - xyloc[1]) < self.half_width) & (xyloc[1] >= ymin)
        elif self.deg == 135:
            sense_fn = lambda xyloc: (np.abs(-xyloc[0] - xyloc[1]) < self.half_width) & (xyloc[1] >= ymin)
        elif self.deg == 0:
            sense_fn = lambda xyloc: (xyloc[1] >= ymin) & (xyloc[1] <= 2 * self.half_width)
        elif self.deg == 'T':
            def sense_fn(xyloc):
                cond0 = (np.abs(xyloc[0]) < self.half_width) & (xyloc[1] >= ymin) & (xyloc[1] <= 250)
                cond1 = (xyloc[1] >= 250) & (xyloc[1] <= (250 + 2 * self.half_width))
                return cond0 | cond1
        else:
            raise ValueError(f'{self.deg} not implemented.')
        self.sense_fn = sense_fn
        self.env_name = f'edge{deg}'
        if np.isfinite(ymin):
            self.env_name += 'ymin' + str(ymin).replace('-', 'b')

    def sense(self, xyloc=None, **kwargs):
        odor = self.sense_fn(xyloc)
        return odor, self.wind, self.mode


class EnvJump(EnvBase):
    # a jumping vertical edge that jumps away from the agent everytime it exits
    def __init__(self, half_width=25, jump_step=20):
        super(EnvJump, self).__init__()
        self.half_width = half_width
        self.jump_step = jump_step
        self.env_edge = EnvEdge(deg=90, half_width=half_width)
        self.env_name = 'jump90'

    def sense(self, it=None, xyloc=None, odors=None, steps=None, **kwargs):
        stim_diff = np.diff(odors[:it])
        iexit = np.argwhere(stim_diff == -1).flatten()
        xsign = np.sign([steps[i, 0] for i in iexit])
        offset = sum(xsign) * self.jump_step
        odor = self.env_edge.sense(xyloc=xyloc + np.array([offset, 0]))[0]
        return odor, self.wind, self.mode


class EnvRecorded(EnvBase):
    def __init__(self, stims_recorded=None):
        super(EnvRecorded, self).__init__()
        self.stims_recorded = stims_recorded
        self.env_name = 'recorded'

    def sense(self, it=None, **kwargs):
        return self.stims_recorded[it], self.wind, self.mode


class EnvReplay(EnvBase):
    # 0~T1: edge tracking; T1~T2: clean air; T2~: replay odor experience from 0~T1
    def __init__(self, deg=None, half_width=25, T1=None, T2=None, dt=None):
        super(EnvReplay, self).__init__()
        self.env_edge = EnvEdge(deg=deg, half_width=half_width)
        self.T1 = T1
        self.T2 = T2
        self.dt = dt
        self.env_name = f'replay{deg}'

    def sense(self, it=None, xyloc=None, odors=None, **kwargs):
        if it * self.dt < self.T1:
            mode = 0
            odor = self.env_edge.sense(xyloc=xyloc)[0]
        elif self.T1 <= it * self.dt < self.T2:
            mode = 1
            odor = 0
        else:
            mode = 2
            odor = odors[it - int(self.T2 / self.dt)]
        return odor, self.wind, mode


class EnvDirection(EnvBase):
    # reinforce exit or entry directions
    def __init__(self, mimic_deg=None, mimic_side=None, dt=None, dur_thres=1.0, dist_thres=3.0):
        super(EnvDirection, self).__init__()
        self.dt = dt
        self.dur_thres = dur_thres
        self.dist_thres = dist_thres
        if mimic_deg == 45:     # mimic 45 degree edge tracking
            self.rng_exit = lambda angl: np.pi / 4 <= angl <= 3 * np.pi / 4
            self.rng_entr = lambda angl: -np.pi / 4 <= angl <= 0
        elif mimic_deg == 135:      # mimic 135 degree edge tracking
            self.rng_exit = lambda angl: np.pi / 4 <= angl <= 3 * np.pi / 4
            self.rng_entr = lambda angl: -np.pi <= angl <= - 3 * np.pi / 4
        elif (mimic_deg == 90) and (mimic_side == 'left'):      # mimic 90 degree left side edge tracking
            self.rng_exit = lambda angl: np.pi / 2 <= angl <= 3 * np.pi / 4
            self.rng_entr = lambda angl: -np.pi / 4 <= angl <= np.pi / 4
        elif (mimic_deg == 90) and (mimic_side == 'right'):     # mimic 90 degree right side edge tracking
            self.rng_exit = lambda angl: np.pi / 4 <= angl <= np.pi / 2
            self.rng_entr = lambda angl: (angl <= -3 * np.pi / 4) | (angl >= 3 * np.pi / 4)
        self.env_name = f'drct{mimic_deg}{mimic_side}'

    def sense(self, it=None, ichng=None, odors=None, steps=None, **kwargs):
        disp = steps[(it - int(self.dur_thres / self.dt)):it].sum(axis=0)   # displacement during the past few seconds
        disp_angl = np.arctan2(disp[1], disp[0])    # angle of the displacement
        disp_dist = np.linalg.norm(disp)    # distance of the displacement
        dt_chng = (it - ichng) * self.dt    # time since last odor change
        rng_fn = self.rng_exit if odors[it - 1] else self.rng_entr      # angle range used for reinforcement
        cross = (disp_dist > self.dist_thres) & (dt_chng > self.dur_thres) & rng_fn(disp_angl)
        odor = (1 - odors[it - 1]) if cross else odors[it - 1]
        return odor, self.wind, self.mode


class EnvAlternateDirection(EnvBase):
    # alternate between different reinforced directions. 0~T1: clean air; T1~T2: direction 1; T2~: direction 2
    def __init__(self, mimic_deg_list=[None, None], mimic_side_list=[None, None], T1=None, T2=None, dt=None,
                 dur_thres=1.0, dist_thres=3.0):
        super(EnvAlternateDirection, self).__init__()
        self.T1 = T1
        self.T2 = T2
        self.dt = dt
        self.dur_thres = dur_thres
        self.dist_thres = dist_thres
        self.env_direction_list = [EnvDirection(mimic_deg=arg1, mimic_side=arg2, dt=dt,
                                                dur_thres=dur_thres, dist_thres=dist_thres)
                                   for arg1, arg2 in zip(mimic_deg_list, mimic_side_list)]
        self.env_name = 'drct' + 'a'.join([str(cur) for cur in mimic_deg_list])

    def sense(self, it=None, ichng=None, odors=None, steps=None, **kwargs):
        if it * self.dt < self.T1:
            mode = 0
            odor = 0
        elif self.T1 < it * self.dt < self.T2:
            mode = 1
            odor = self.env_direction_list[0].sense(it=it, ichng=ichng, odors=odors, steps=steps)[0]
        else:
            mode = 2
            odor = self.env_direction_list[1].sense(it=it, ichng=ichng, odors=odors, steps=steps)[0]
        return odor, self.wind, mode


class EnvZigzag(EnvBase):
    def __init__(self, deg1=None, deg2=None, deg3=None, y1=None, nx=None, dt=None,
                 dur_thres=1.0, dist_thres=3.0):
        super(EnvZigzag, self).__init__()
        self.env_edge1 = EnvEdge(deg=deg1, half_width=25*np.sqrt(2))
        self.env_direction = EnvDirection(mimic_deg=deg2, dt=dt, dur_thres=dur_thres, dist_thres=dist_thres)
        self.env_edge2 = EnvEdge(deg=deg3, half_width=25*np.sqrt(2), ymin=-20)
        self.y1 = y1
        self.x1 = y1 / np.tan(deg1 / 180 * np.pi)
        self.nx = nx      # number of crossings (exit + entry) during reinforcement
        self.it1 = None     # the time step when agent finished tracking edge 1
        self.it2 = None      # the time step when agent have just crossed 'nx' times during reinforcement phase
        self.dt = dt
        self.dur_thres = dur_thres
        self.env_name = f'zigzagH{y1}a{deg1}r{deg2}a{deg3}nx{nx}'

    def reset(self):
        self.it1 = None
        self.it2 = None      # reset for every simulation
        return

    def sense_base(self, xyloc):
        if xyloc[1] <= self.y1:
            mode = 0
            odor = self.env_edge1.sense(xyloc=xyloc)[0]
        else:
            mode = 1
            odor = self.env_edge2.sense(xyloc=[xyloc[0] - self.x1, xyloc[1] - self.y1])[0]
        return odor, self.wind, mode

    def sense(self, xyloc=None, it=None, ichng=None, odors=None, steps=None, xylocs=None, **kwargs):
        if self.nx is None:
            return self.sense_base(xyloc)

        if self.it1 is None:
            mode = 0
            odor = self.env_edge1.sense(xyloc=xyloc)[0]
            if (odor == 0) and (odors[it - 1] == 1) and (xyloc[1] >= self.y1):
                self.it1 = it
                if self.nx == 0:
                    self.it2 = it
                    odor = 1
        elif self.it2 is None:
            mode = 1
            odor = self.env_direction.sense(it=it, ichng=ichng, odors=odors, steps=steps)[0]
            icross = np.argwhere(np.diff(np.append(odors[self.it1:it], odor))).flatten()
            if len(icross) == self.nx:  # nx should an odd number: 2*num_entry -1. (entry-exit-entry)
                self.it2 = it
        else:
            mode = 2
            odor = self.env_edge2.sense(xyloc=xyloc - xylocs[self.it2])[0]
        return odor, self.wind, mode