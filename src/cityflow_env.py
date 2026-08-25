#za da moze da se tretira sekoja raskrsnica kako agent
import json
import os
import numpy as np
import cityflow

class CityFlowEnv:
    def __init__(self, config_path, roadnet_path, episode_steps=3600, #pateka do conf.json, roadnet.json, na klk cekori po epzioda, vremetraenje, threads
                 step_length_per_action=10, thread_num=1):

        self.eng = self._create_engine(config_path, thread_num) #simulator
        self.episode_steps = episode_steps # klk simulation steps trae edna epizoda
        self.step_length_per_action = step_length_per_action # klk simulacii po izbrana odluka

        with open(roadnet_path) as f:
            self.roadnet = json.load(f) #roadnet.json file

        self._build_topology()

        self.n_agents = len(self.inter_ids)
        self.obs_dim = self._obs_dim()  # golemina na observation vectorot
        self.action_dims = [len(self.phases[iid]) for iid in self.inter_ids] #klk akcii ima sekoj agent
        self.max_actions = max(self.action_dims)

        self.current_time = 0


    @staticmethod
    def _create_engine(config_path, thread_num):
        #za pateki vo config.json
        #patekite se relativni na working dir ne na config.json

        config_path = os.path.abspath(config_path)
        config_dir = os.path.dirname(config_path)

        with open(config_path) as f:
            cfg = json.load(f)

        sub_dir = cfg.get("dir", "./")
        resolved_dir = os.path.normpath(os.path.join(config_dir, sub_dir))

        for key in ("roadnetFile", "flowFile"):
            fname = cfg.get(key)
            if not fname:
                continue
            full_path = os.path.join(resolved_dir, fname)
            if not os.path.exists(full_path):
                raise FileNotFoundError()

        old_cwd = os.getcwd()
        os.chdir(config_dir)
        try:
            eng = cityflow.Engine(os.path.basename(config_path), thread_num=thread_num)
        finally:
            os.chdir(old_cwd)
        return eng

    #parsiranje na raskrsnici
    def _build_topology(self):
        roads_by_id = {r["id"]: r for r in self.roadnet["roads"]}

        self.inter_ids = []  #niza id-a na raskrsnici
        self.in_lanes = {}   # za raskrsnica koi lanes doagjaat
        self.phases = {}     # za raskrsnica fazi na svetla na semaforite

        for inter in self.roadnet["intersections"]:
            if inter.get("virtual", False):
                continue
            iid = inter["id"]

            incoming_roads = set()
            for rl in inter.get("roadLinks", []):
                incoming_roads.add(rl["startRoad"])

            lanes = []
            for road_id in incoming_roads:
                road = roads_by_id.get(road_id)
                if road is None:
                    continue
                n_lanes = len(road["lanes"])
                lanes.extend(f"{road_id}_{i}" for i in range(n_lanes))

            self.in_lanes[iid] = sorted(lanes)

            tl = inter.get("trafficLight", {})
            n_phases = len(tl.get("lightphases", []))
            self.phases[iid] = list(range(max(n_phases, 1)))

            self.inter_ids.append(iid)

        self.inter_ids.sort()

    def _obs_dim(self):
        max_lanes = max(len(v) for v in self.in_lanes.values())
        max_phases = max(len(v) for v in self.phases.values())
        self._max_lanes = max_lanes
        self._max_phases = max_phases
        return max_lanes * 2 + max_phases

    def reset(self):
        self.eng.reset()
        self.current_time = 0
        self._current_phase = {iid: 0 for iid in self.inter_ids}
        return self._get_obs()


    def step(self, actions): # primenuvanje na akcijata koja ja prezel sekoj od agentite(raskrsnici)

        if isinstance(actions, (list, np.ndarray)):
            actions = {iid: int(a) for iid, a in zip(self.inter_ids, actions)}

        for iid, phase in actions.items():
            phase = min(phase, len(self.phases[iid]) - 1)  # clip to valid range
            self.eng.set_tl_phase(iid, phase)
            self._current_phase[iid] = phase

        for _ in range(self.step_length_per_action):
            self.eng.next_step()
            self.current_time += 1

        obs = self._get_obs()
        reward = self._get_shared_reward()
        done = self.current_time >= self.episode_steps
        info = {"avg_travel_time": self.eng.get_average_travel_time()}

        return obs, self.get_global_state(obs), reward, done, info

    def _get_obs(self): # za momentalna sostojba, lokalen observation za sekoja raskrsnica
        vcount = self.eng.get_lane_vehicle_count()
        wcount = self.eng.get_lane_waiting_vehicle_count()

        obs = {}
        for iid in self.inter_ids:
            lane_feats = []
            for lane in self.in_lanes[iid]:
                lane_feats.append(vcount.get(lane, 0))
                lane_feats.append(wcount.get(lane, 0))
            pad = (self._max_lanes * 2) - len(lane_feats)
            lane_feats.extend([0] * pad)

            phase_onehot = [0] * self._max_phases
            phase_onehot[self._current_phase[iid]] = 1

            obs[iid] = np.array(lane_feats + phase_onehot, dtype=np.float32)
        return obs

    def get_global_state(self, obs): # site local pogledi konkatenirani
        return np.concatenate([obs[iid] for iid in self.inter_ids])

    def _get_shared_reward(self): #zaednicka nagrada za site agenti
        wcount = self.eng.get_lane_waiting_vehicle_count()
        total_waiting = sum(wcount.values())
        return -float(total_waiting) / max(self.n_agents, 1)