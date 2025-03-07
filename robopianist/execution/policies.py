import os
import pickle
import threading
import tkinter as tk
from typing import Tuple

import dm_env
import numpy as np
import requests
from dm_env_wrappers._src import canonical_spec
from flax.serialization import from_bytes

from configs import Defaults
from sac import SAC


class Policy:
    def __init__(self, env: dm_env.Environment, **kwargs) -> None:
        pass

    def reset(self):
        pass

    def __call__(self, timestep: dm_env.TimeStep) -> np.ndarray:
        raise NotImplementedError


class NullPolicy(Policy):
    def __init__(self, env: dm_env.Environment, **kwargs) -> None:
        super().__init__(env)
        action_spec = env.action_spec()
        self.zeros = np.zeros(action_spec.shape, dtype=action_spec.dtype)

    def __call__(self, timestep: dm_env.TimeStep = None, observations=None) -> np.ndarray:
        return self.zeros


class AlternatePolicy(Policy):
    def __init__(self, env: dm_env.Environment, steps=60, **kwargs) -> None:
        super().__init__(env)
        self._idx = steps // 2
        mins = env.action_spec().minimum
        maxes = env.action_spec().maximum
        self._actions = np.linspace(mins, maxes, steps)
        self._actions = np.concatenate([self._actions, np.flip(self._actions, 0)])

    def reset(self):
        self._idx = 0

    def __call__(self, timestep=None, **kwargs) -> np.ndarray:
        action = self._actions[self._idx]
        self._idx += 1
        self._idx %= len(self._actions)
        return action


class SliderPolicy(Policy):
    def __init__(self, env: dm_env.Environment, **kwargs) -> None:
        super().__init__(env)
        action_spec = env.action_spec()
        self.actions = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        self.actions_reset = np.zeros(action_spec.shape, dtype=action_spec.dtype)

        self.sliders_thread = threading.Thread(target=self.init_tk, args=(action_spec, env.task.action_labels),
                                               daemon=True)
        self.sliders_thread.start()

    def init_tk(self, action_space, names):
        def create_callback(i):
            return lambda val: update_val(i, val)

        def update_val(index, val):
            self.actions[index] = val

        master = tk.Tk()
        font = ("Arial", 25)
        colls = 2
        for i, (min, max, name) in enumerate(zip(action_space.minimum, action_space.maximum, names)):
            coll, row = (i % colls) * 2, i // colls

            tk.Label(master, text=f"{i}: {name}", font=font).grid(column=coll, row=row)
            tk.Scale(master, from_=min, to=max, orient=tk.HORIZONTAL, font=font, resolution=0.01,
                     command=create_callback(i), length=600, width=35, name=f"{i}").grid(column=coll + 1, row=row)
        i = len(names)
        coll, row = (i % colls) * 2, i // colls

        def set():
            self.actions_reset = self.actions.copy()

        def reset():
            self.actions = self.actions_reset.copy()

        set()
        tk.Button(master, command=set, text="SET", font=font).grid(column=coll, row=row, sticky=tk.NSEW, padx=5,
                                                                   pady=5)
        tk.Button(master, command=reset, text="RESET", font=font).grid(column=coll + 1, row=row, sticky=tk.NSEW,
                                                                       padx=5, pady=5)
        tk.mainloop()

    def __call__(self, timestep: dm_env.TimeStep = None, **kwargs) -> np.ndarray:
        if "observations" in kwargs.keys():
            key_list = ["A" if x else "-" for x in kwargs["observations"]["piano/activation"]]
            print("Pressed: " + "".join(key_list))
        return self.actions


class ModelPolicy(Policy):
    def __init__(self, env: dm_env.Environment, args: Defaults, **kwargs) -> None:
        super().__init__(env)

        self.args = args
        self.action_spec = env.action_spec()

        env_path, model_path = download_model(args.model_url)
        env_spec = pickle.load(open(env_path, "rb"))
        self.model = SAC.initialize(spec=env_spec, config=args.agent_config, seed=args.seed, discount=args.discount)

        with open(model_path, "rb") as file:
            self.model = from_bytes(self.model, file.read())

    def __call__(self, timestep: dm_env.TimeStep = None, observations=None, verbose=False) -> np.ndarray:
        if observations is None:
            observations = timestep.observation

        goal = observations["goal"]
        pressed = observations["piano/activation"]
        hand_joints = observations["lh_allegro_hand/joints_pos"]
        if verbose:
            key_list = ["A" if x else "-" for x in pressed]
            print("Pressed: " + "".join(key_list))
            key_list = ["A" if x == 1.0 else "-" for x in goal]
            print("Goal: " + "".join(key_list))
            print("Joints: ", end="")
            np.set_printoptions(precision=3, linewidth=32)
            print(hand_joints)
            print("\n")
        observations = np.concatenate([goal, hand_joints, pressed])
        actions = self.model.eval_actions(observations)

        if self.args.canonicalize:
            actions = canonical_spec._scale_nested_action(actions, self.action_spec, self.args.clip)
        return actions


def download_model(model_url: str) -> Tuple[str, str]:
    def _prepare_cache(path: str, url: str):
        if not os.path.exists(path):
            print(f"download model \nfrom {url}\nto {path}")
            os.makedirs(path, exist_ok=True)
            os.rmdir(path)
            response = requests.get(url)
            with open(path, "wb") as file:
                file.write(response.content)
        else:
            print(f"use cached model file at {path}")

    model_path = "/tmp/" + model_url[model_url.find("robopianist"):]
    last_slash = model_url.rfind("/")
    env_url = model_url[:last_slash + 1] + "env_spec.pkl"
    env_path = "/tmp/" + env_url[env_url.find("robopianist"):]

    _prepare_cache(path=env_path, url=env_url)
    _prepare_cache(path=model_path, url=model_url)

    return env_path, model_path
