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
}
