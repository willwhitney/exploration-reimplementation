import numpy as np

from collections import OrderedDict
from dm_env.specs import BoundedArray


DOMAINS = {
    'point_mass': {
        'easy': OrderedDict({
            'position': BoundedArray(name='position', shape=(2,),
                                     dtype=np.float32,
                                     minimum=np.array([-0.3, -0.3]),
                                     maximum=np.array([0.3, 0.3])),
            'velocity': BoundedArray(name='velocity', shape=(2,),
                                     dtype=np.float32,
                                     minimum=np.array([-0.5, -0.5]),
                                     maximum=np.array([0.5, 0.5])),
        }),
    },
    'point': {
        'velocity': OrderedDict({
            'position': BoundedArray(name='position', shape=(2,),
                                     dtype=np.float32,
                                     minimum=np.array([-0.3, -0.3]),
                                     maximum=np.array([0.3, 0.3])),
        }),
        'velocity_slow': OrderedDict({
            'position': BoundedArray(name='position', shape=(2,),
                                     dtype=np.float32,
                                     minimum=np.array([-0.3, -0.3]),
                                     maximum=np.array([0.3, 0.3])),
        }),
        'mass': OrderedDict({
            'position': BoundedArray(name='position', shape=(2,),
                                     dtype=np.float32,
                                     minimum=np.array([-0.3, -0.3]),
                                     maximum=np.array([0.3, 0.3])),
            'velocity': BoundedArray(name='velocity', shape=(2,),
                                     dtype=np.float32,
                                     minimum=np.array([-0.5, -0.5]),
                                     maximum=np.array([0.5, 0.5])),
        }),
    },
    'cartpole': {
        'swingup': OrderedDict({
            'position': BoundedArray(name='position', shape=(3,),
                                     dtype=np.float32,
                                     minimum=np.array([-2, -1, -1]),
                                     maximum=np.array([2, 1, 1])),
            'velocity': BoundedArray(name='velocity', shape=(2,),
                                     dtype=np.float32,
                                     minimum=np.array([-5, -5]),
                                     maximum=np.array([5, 5])),
        }),
        'swingup_sparse': OrderedDict({
            'position': BoundedArray(name='position', shape=(3,),
                                     dtype=np.float32,
                                     minimum=np.array([-2, -1, -1]),
                                     maximum=np.array([2, 1, 1])),
            'velocity': BoundedArray(name='velocity', shape=(2,),
                                     dtype=np.float32,
                                     minimum=np.array([-5, -5]),
                                     maximum=np.array([5, 5])),
        }),
        'balance': OrderedDict({
            'position': BoundedArray(name='position', shape=(3,),
                                     dtype=np.float32,
                                     minimum=np.array([-2, -1, -1]),
                                     maximum=np.array([2, 1, 1])),
            'velocity': BoundedArray(name='velocity', shape=(2,),
                                     dtype=np.float32,
                                     minimum=np.array([-5, -5]),
                                     maximum=np.array([5, 5])),
        }),
    },
    'hopper': {
        'hop': OrderedDict({
            'position': BoundedArray(name='position', shape=(6,),
                                     dtype=np.float32,
                                     minimum=np.array([-1, -5, -1, -5, -5, -1]),
                                     maximum=np.array([0, 5, 1, 5, 5, 1])),
            'velocity': BoundedArray(name='velocity', shape=(7,),
                                     dtype=np.float32,
                                     minimum=np.array([-5, -10, -20, -20, -20, -20, -20]),
                                     maximum=np.array([5, 10, 20, 20, 20, 20, 20])),
            'touch': BoundedArray(name='touch', shape=(2,),
                                  dtype=np.float32,
                                  minimum=np.array([0, 0]),
                                  maximum=np.array([10, 10])),
        }),
    },
    'acrobot': {
        'swingup': OrderedDict({
            'orientations': BoundedArray(name='orientations', shape=(4,),
                                     dtype=np.float32,
                                     minimum=np.array([-1, -1, -1, -1]),
                                     maximum=np.array([1, 1, 1, 1])),
            'velocity': BoundedArray(name='velocity', shape=(2,),
                                     dtype=np.float32,
                                     minimum=np.array([-20, -20]),
                                     maximum=np.array([20, 20])),
        }),
    }
}


if __name__ == '__main__':
    import jax
    from dm_control import suite
    import utils

    env_name = 'hopper'
    task_name = 'hop'
    env = suite.load(env_name, task_name)
    ospec = env.observation_spec()
    aspec = env.action_spec()

    policies = [
        lambda: np.random.uniform(low=aspec.minimum, high=aspec.maximum),
        lambda: aspec.minimum,
        lambda: aspec.maximum,
    ]

    observations = []

    for policy in policies:
        timestep = env.reset()
        for i in range(10000):
            observations.append(timestep.observation)
            a = policy()
            timestep = env.step(a)

    obs_stack = utils.tree_stack(observations)
    print("Min values:")
    print(jax.tree_map(lambda x: x.min(axis=0),
                       obs_stack))
    print("Max values:")
    print(jax.tree_map(lambda x: x.max(axis=0),
                       obs_stack))
