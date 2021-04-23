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
    'ball_in_cup': {
        'catch': OrderedDict({
            'position': BoundedArray(name='position', shape=(4,),
                                     dtype=np.float32,
                                     minimum=np.array([-0.25, -0.3, -0.2, -0.2]),
                                     maximum=np.array([0.25, 0.2, 0.5, 0.25])),
            'velocity': BoundedArray(name='velocity', shape=(4,),
                                     dtype=np.float32,
                                     minimum=np.array([-1.5, -1.5, -1.5, -3.0]),
                                     maximum=np.array([1.5, 1.5, 1.5, 1])),
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
    },
    'manipulator': {
        'bring_ball': OrderedDict({
            'arm_pos': BoundedArray(name='arm_pos', shape=(8, 2),
                                     dtype=np.float32,
                                     minimum=-np.ones((8, 2)),
                                     maximum=np.ones((8, 2)),),
            'arm_vel': BoundedArray(name='arm_vel', shape=(8,),
                                     dtype=np.float32,
                                     minimum=-10 * np.ones((8,)),
                                     maximum=10 * np.ones((8,)),),
            'touch': BoundedArray(name='touch', shape=(5,),
                                     dtype=np.float32,
                                     minimum=np.zeros((5,)),
                                     maximum=5 * np.ones((5,)),),
            'hand_pos': BoundedArray(name='hand_pos', shape=(4,),
                                     dtype=np.float32,
                                     minimum=-np.ones((4,)),
                                     maximum=np.ones((4,)),),
            'object_pos': BoundedArray(name='object_pos', shape=(4,),
                                     dtype=np.float32,
                                     minimum=-np.ones((4,)),
                                     maximum=np.ones((4,)),),
            'object_vel': BoundedArray(name='object_vel', shape=(3,),
                                     dtype=np.float32,
                                     minimum=-10 * np.ones((3,)),
                                     maximum=10 * np.ones((3,)),),
            'target_pos': BoundedArray(name='target_pos', shape=(4,),
                                     dtype=np.float32,
                                     minimum=-np.ones((4,)),
                                     maximum=np.ones((4,)),),
        }),
    },
    'finger': {
        'turn_easy': OrderedDict({
            'position': BoundedArray(name='position', shape=(4,),
                                     dtype=np.float32,
                                     minimum=np.array([-2, -2, -.1, -.1]),
                                     maximum=np.array([2, 2, .1, .1])),
            'velocity': BoundedArray(name='velocity', shape=(3,),
                                     dtype=np.float32,
                                     minimum=-12 * np.ones((3,)),
                                     maximum=12 * np.ones((3,))),
            'touch': BoundedArray(name='touch', shape=(2,),
                                     dtype=np.float32,
                                     minimum=np.zeros((2,)),
                                     maximum=5 * np.ones((2,))),
            'target_position': BoundedArray(name='target_position', shape=(2,),
                                     dtype=np.float32,
                                     minimum=np.array([-0.13, -0.13]),
                                     maximum=np.array([0.13, 0.13])),
            'dist_to_target': BoundedArray(name='dist_to_target', shape=(1,),
                                     dtype=np.float32,
                                     minimum=np.array([-0.2]),
                                     maximum=np.array([0.23])),
        }),
        'turn_hard': OrderedDict({
            'position': BoundedArray(name='position', shape=(4,),
                                     dtype=np.float32,
                                     minimum=np.array([-2, -2, -.1, -.1]),
                                     maximum=np.array([2, 2, .1, .1])),
            'velocity': BoundedArray(name='velocity', shape=(3,),
                                     dtype=np.float32,
                                     minimum=-12 * np.ones((3,)),
                                     maximum=12 * np.ones((3,))),
            'touch': BoundedArray(name='touch', shape=(2,),
                                     dtype=np.float32,
                                     minimum=np.zeros((2,)),
                                     maximum=5 * np.ones((2,))),
            'target_position': BoundedArray(name='target_position', shape=(2,),
                                     dtype=np.float32,
                                     minimum=np.array([-0.13, -0.13]),
                                     maximum=np.array([0.13, 0.13])),
            'dist_to_target': BoundedArray(name='dist_to_target', shape=(1,),
                                     dtype=np.float32,
                                     minimum=np.array([-0.2]),
                                     maximum=np.array([0.23])),
        }),
    },
    'walker': {
        'stand': OrderedDict({
            'orientations': BoundedArray(name='orientations', shape=(14,),
                                     dtype=np.float32,
                                     minimum=-np.ones((14,)),
                                     maximum=np.ones((14,))),
            'height': BoundedArray(name='height', shape=(1,),
                                   dtype=np.float32,
                                   minimum=np.array([0.3,]),
                                   maximum=np.array([1.3,])),
            'velocity': BoundedArray(name='velocity', shape=(9,),
                                   dtype=np.float32,
                                   minimum=np.array([-5, -5, -20, -40, -40, -40, -40, -40, -40]),
                                   maximum=np.array([5, 5, 20, 40, 40, 40, 40, 40, 40])),
        }),
        'walk': OrderedDict({
            'orientations': BoundedArray(name='orientations', shape=(14,),
                                     dtype=np.float32,
                                     minimum=-np.ones((14,)),
                                     maximum=np.ones((14,))),
            'height': BoundedArray(name='height', shape=(1,),
                                   dtype=np.float32,
                                   minimum=np.array([0.3,]),
                                   maximum=np.array([1.3,])),
            'velocity': BoundedArray(name='velocity', shape=(9,),
                                   dtype=np.float32,
                                   minimum=np.array([-5, -5, -20, -40, -40, -40, -40, -40, -40]),
                                   maximum=np.array([5, 5, 20, 40, 40, 40, 40, 40, 40])),
        }),
    },


    # ----------- Custom envs -----------------------------------------
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
    'reacher_explore': {
        'easy': OrderedDict({
            'position': BoundedArray(name='position', shape=(2,),
                                     dtype=np.float32,
                                     minimum=np.array([0, -3]),
                                     maximum=np.array([6.28, 3])),
            'to_target': BoundedArray(name='to_target', shape=(2,),
                                     dtype=np.float32,
                                     minimum=np.array([-0.4, -0.4]),
                                     maximum=np.array([0.4, 0.4])),
            'velocity': BoundedArray(name='velocity', shape=(2,),
                                     dtype=np.float32,
                                     minimum=np.array([-5, -5]),
                                     maximum=np.array([5, 5])),
        }),
        'hard': OrderedDict({
            'position': BoundedArray(name='position', shape=(2,),
                                     dtype=np.float32,
                                     minimum=np.array([0, -3]),
                                     maximum=np.array([6.28, 3])),
            'to_target': BoundedArray(name='to_target', shape=(2,),
                                     dtype=np.float32,
                                     minimum=np.array([-0.4, -0.4]),
                                     maximum=np.array([0.4, 0.4])),
            'velocity': BoundedArray(name='velocity', shape=(2,),
                                     dtype=np.float32,
                                     minimum=np.array([-5, -5]),
                                     maximum=np.array([5, 5])),
        }),
        'hard_fixed_init': OrderedDict({
            'position': BoundedArray(name='position', shape=(2,),
                                     dtype=np.float32,
                                     minimum=np.array([0, -3]),
                                     maximum=np.array([6.28, 3])),
            'to_target': BoundedArray(name='to_target', shape=(2,),
                                     dtype=np.float32,
                                     minimum=np.array([-0.4, -0.4]),
                                     maximum=np.array([0.4, 0.4])),
            'velocity': BoundedArray(name='velocity', shape=(2,),
                                     dtype=np.float32,
                                     minimum=np.array([-5, -5]),
                                     maximum=np.array([5, 5])),
        }),
    },
    'hallway': {
        'velocity_1': OrderedDict({
            'position': BoundedArray(name='position', shape=(2,),
                                     dtype=np.float32,
                                     minimum=np.array([-0.5, -0.1]),
                                     maximum=np.array([0.5, 0.1])),
        }),
        'velocity_4': OrderedDict({
            'position': BoundedArray(name='position', shape=(2,),
                                     dtype=np.float32,
                                     minimum=np.array([-2, -0.1]),
                                     maximum=np.array([2, 0.1])),
        }),
        'velocity_1_distractor': OrderedDict({
            'position': BoundedArray(name='position', shape=(2,),
                                     dtype=np.float32,
                                     minimum=np.array([-0.5, -0.1]),
                                     maximum=np.array([0.5, 0.1])),
        }),
        'velocity_4_distractor': OrderedDict({
            'position': BoundedArray(name='position', shape=(2,),
                                     dtype=np.float32,
                                     minimum=np.array([-2, -0.1]),
                                     maximum=np.array([2, 0.1])),
        }),
        'velocity_4_inverse_distractor': OrderedDict({
            'position': BoundedArray(name='position', shape=(2,),
                                     dtype=np.float32,
                                     minimum=np.array([-2, -0.1]),
                                     maximum=np.array([2, 0.1])),
        }),
    },
    'ball_in_cup_explore': {
        'catch': OrderedDict({
            'position': BoundedArray(name='position', shape=(4,),
                                     dtype=np.float32,
                                     minimum=np.array([-0.25, -0.3, -0.2, -0.2]),
                                     maximum=np.array([0.25, 0.2, 0.5, 0.25])),
            'velocity': BoundedArray(name='velocity', shape=(4,),
                                     dtype=np.float32,
                                     minimum=np.array([-1.5, -1.5, -1.5, -3.0]),
                                     maximum=np.array([1.5, 1.5, 1.5, 1])),
        }),
    },
    'manipulator_explore': {
        'reach_lift_ball': OrderedDict({
            'arm_pos': BoundedArray(name='arm_pos', shape=(8, 2),
                                     dtype=np.float32,
                                     minimum=-np.ones((8, 2)),
                                     maximum=np.ones((8, 2)),),
            'arm_vel': BoundedArray(name='arm_vel', shape=(8,),
                                     dtype=np.float32,
                                     minimum=-10 * np.ones((8,)),
                                     maximum=10 * np.ones((8,)),),
            'touch': BoundedArray(name='touch', shape=(5,),
                                     dtype=np.float32,
                                     minimum=np.zeros((5,)),
                                     maximum=5 * np.ones((5,)),),
            'hand_pos': BoundedArray(name='hand_pos', shape=(4,),
                                     dtype=np.float32,
                                     minimum=-np.ones((4,)),
                                     maximum=np.ones((4,)),),
            'object_pos': BoundedArray(name='object_pos', shape=(4,),
                                     dtype=np.float32,
                                     minimum=np.array([-0.4, 0, -1, 0]),
                                     maximum=np.array([0.4, 1, 1, 1]),),
            'object_vel': BoundedArray(name='object_vel', shape=(3,),
                                     dtype=np.float32,
                                     minimum=-10 * np.ones((3,)),
                                     maximum=10 * np.ones((3,)),),
        }),
        'reach_lift_ball_narrow': OrderedDict({
            'arm_pos': BoundedArray(name='arm_pos', shape=(8, 2),
                                     dtype=np.float32,
                                     minimum=-np.ones((8, 2)),
                                     maximum=np.ones((8, 2)),),
            'arm_vel': BoundedArray(name='arm_vel', shape=(8,),
                                     dtype=np.float32,
                                     minimum=-10 * np.ones((8,)),
                                     maximum=10 * np.ones((8,)),),
            'touch': BoundedArray(name='touch', shape=(5,),
                                     dtype=np.float32,
                                     minimum=np.zeros((5,)),
                                     maximum=5 * np.ones((5,)),),
            'hand_pos': BoundedArray(name='hand_pos', shape=(4,),
                                     dtype=np.float32,
                                     minimum=-np.ones((4,)),
                                     maximum=np.ones((4,)),),
            'object_pos': BoundedArray(name='object_pos', shape=(4,),
                                     dtype=np.float32,
                                     minimum=np.array([-0.4, 0, -1, 0]),
                                     maximum=np.array([0.4, 1, 1, 1]),),
            'object_vel': BoundedArray(name='object_vel', shape=(3,),
                                     dtype=np.float32,
                                     minimum=-10 * np.ones((3,)),
                                     maximum=10 * np.ones((3,)),),
        }),
        'reach_lift_ball_fixed': OrderedDict({
            'arm_pos': BoundedArray(name='arm_pos', shape=(8, 2),
                                     dtype=np.float32,
                                     minimum=-np.ones((8, 2)),
                                     maximum=np.ones((8, 2)),),
            'arm_vel': BoundedArray(name='arm_vel', shape=(8,),
                                     dtype=np.float32,
                                     minimum=-10 * np.ones((8,)),
                                     maximum=10 * np.ones((8,)),),
            'touch': BoundedArray(name='touch', shape=(5,),
                                     dtype=np.float32,
                                     minimum=np.zeros((5,)),
                                     maximum=5 * np.ones((5,)),),
            'hand_pos': BoundedArray(name='hand_pos', shape=(4,),
                                     dtype=np.float32,
                                     minimum=-np.ones((4,)),
                                     maximum=np.ones((4,)),),
            'object_pos': BoundedArray(name='object_pos', shape=(4,),
                                     dtype=np.float32,
                                     minimum=np.array([-0.4, 0, -1, 0]),
                                     maximum=np.array([0.4, 1, 1, 1]),),
            'object_vel': BoundedArray(name='object_vel', shape=(3,),
                                     dtype=np.float32,
                                     minimum=-10 * np.ones((3,)),
                                     maximum=10 * np.ones((3,)),),
        }),
        'reach_shaped_lift_ball': OrderedDict({
            'arm_pos': BoundedArray(name='arm_pos', shape=(8, 2),
                                     dtype=np.float32,
                                     minimum=-np.ones((8, 2)),
                                     maximum=np.ones((8, 2)),),
            'arm_vel': BoundedArray(name='arm_vel', shape=(8,),
                                     dtype=np.float32,
                                     minimum=-10 * np.ones((8,)),
                                     maximum=10 * np.ones((8,)),),
            'touch': BoundedArray(name='touch', shape=(5,),
                                     dtype=np.float32,
                                     minimum=np.zeros((5,)),
                                     maximum=5 * np.ones((5,)),),
            'hand_pos': BoundedArray(name='hand_pos', shape=(4,),
                                     dtype=np.float32,
                                     minimum=-np.ones((4,)),
                                     maximum=np.ones((4,)),),
            'object_pos': BoundedArray(name='object_pos', shape=(4,),
                                     dtype=np.float32,
                                     minimum=np.array([-0.4, 0, -1, 0]),
                                     maximum=np.array([0.4, 1, 1, 1]),),
            'object_vel': BoundedArray(name='object_vel', shape=(3,),
                                     dtype=np.float32,
                                     minimum=-10 * np.ones((3,)),
                                     maximum=10 * np.ones((3,)),),
        }),
        'reach_shaped_lift_ball_narrow': OrderedDict({
            'arm_pos': BoundedArray(name='arm_pos', shape=(8, 2),
                                     dtype=np.float32,
                                     minimum=-np.ones((8, 2)),
                                     maximum=np.ones((8, 2)),),
            'arm_vel': BoundedArray(name='arm_vel', shape=(8,),
                                     dtype=np.float32,
                                     minimum=-10 * np.ones((8,)),
                                     maximum=10 * np.ones((8,)),),
            'touch': BoundedArray(name='touch', shape=(5,),
                                     dtype=np.float32,
                                     minimum=np.zeros((5,)),
                                     maximum=5 * np.ones((5,)),),
            'hand_pos': BoundedArray(name='hand_pos', shape=(4,),
                                     dtype=np.float32,
                                     minimum=-np.ones((4,)),
                                     maximum=np.ones((4,)),),
            'object_pos': BoundedArray(name='object_pos', shape=(4,),
                                     dtype=np.float32,
                                     minimum=np.array([-0.4, 0, -1, 0]),
                                     maximum=np.array([0.4, 1, 1, 1]),),
            'object_vel': BoundedArray(name='object_vel', shape=(3,),
                                     dtype=np.float32,
                                     minimum=-10 * np.ones((3,)),
                                     maximum=10 * np.ones((3,)),),
        }),
        'reach_shaped_lift_ball_fixed': OrderedDict({
            'arm_pos': BoundedArray(name='arm_pos', shape=(8, 2),
                                     dtype=np.float32,
                                     minimum=-np.ones((8, 2)),
                                     maximum=np.ones((8, 2)),),
            'arm_vel': BoundedArray(name='arm_vel', shape=(8,),
                                     dtype=np.float32,
                                     minimum=-10 * np.ones((8,)),
                                     maximum=10 * np.ones((8,)),),
            'touch': BoundedArray(name='touch', shape=(5,),
                                     dtype=np.float32,
                                     minimum=np.zeros((5,)),
                                     maximum=5 * np.ones((5,)),),
            'hand_pos': BoundedArray(name='hand_pos', shape=(4,),
                                     dtype=np.float32,
                                     minimum=-np.ones((4,)),
                                     maximum=np.ones((4,)),),
            'object_pos': BoundedArray(name='object_pos', shape=(4,),
                                     dtype=np.float32,
                                     minimum=np.array([-0.4, 0, -1, 0]),
                                     maximum=np.array([0.4, 1, 1, 1]),),
            'object_vel': BoundedArray(name='object_vel', shape=(3,),
                                     dtype=np.float32,
                                     minimum=-10 * np.ones((3,)),
                                     maximum=10 * np.ones((3,)),),
        }),
    },
    'walker_explore': {
        'stand_narrow': OrderedDict({
            'orientations': BoundedArray(name='orientations', shape=(14,),
                                     dtype=np.float32,
                                     minimum=-np.ones((14,)),
                                     maximum=np.ones((14,))),
            'height': BoundedArray(name='height', shape=(1,),
                                   dtype=np.float32,
                                   minimum=np.array([0.3,]),
                                   maximum=np.array([1.3,])),
            'velocity': BoundedArray(name='velocity', shape=(9,),
                                   dtype=np.float32,
                                   minimum=np.array([-5, -5, -20, -40, -40, -40, -40, -40, -40]),
                                   maximum=np.array([5, 5, 20, 40, 40, 40, 40, 40, 40])),
        }),
        'stand_narrow_sparse': OrderedDict({
            'orientations': BoundedArray(name='orientations', shape=(14,),
                                     dtype=np.float32,
                                     minimum=-np.ones((14,)),
                                     maximum=np.ones((14,))),
            'height': BoundedArray(name='height', shape=(1,),
                                   dtype=np.float32,
                                   minimum=np.array([0.3,]),
                                   maximum=np.array([1.3,])),
            'velocity': BoundedArray(name='velocity', shape=(9,),
                                   dtype=np.float32,
                                   minimum=np.array([-5, -5, -20, -40, -40, -40, -40, -40, -40]),
                                   maximum=np.array([5, 5, 20, 40, 40, 40, 40, 40, 40])),
        }),
        'walk_narrow_sparse': OrderedDict({
            'orientations': BoundedArray(name='orientations', shape=(14,),
                                     dtype=np.float32,
                                     minimum=-np.ones((14,)),
                                     maximum=np.ones((14,))),
            'height': BoundedArray(name='height', shape=(1,),
                                   dtype=np.float32,
                                   minimum=np.array([0.3,]),
                                   maximum=np.array([1.3,])),
            'velocity': BoundedArray(name='velocity', shape=(9,),
                                   dtype=np.float32,
                                   minimum=np.array([-5, -5, -20, -40, -40, -40, -40, -40, -40]),
                                   maximum=np.array([5, 5, 20, 40, 40, 40, 40, 40, 40])),
        }),
        'run_narrow_sparse': OrderedDict({
            'orientations': BoundedArray(name='orientations', shape=(14,),
                                     dtype=np.float32,
                                     minimum=-np.ones((14,)),
                                     maximum=np.ones((14,))),
            'height': BoundedArray(name='height', shape=(1,),
                                   dtype=np.float32,
                                   minimum=np.array([0.3,]),
                                   maximum=np.array([1.3,])),
            'velocity': BoundedArray(name='velocity', shape=(9,),
                                   dtype=np.float32,
                                   minimum=np.array([-5, -5, -20, -40, -40, -40, -40, -40, -40]),
                                   maximum=np.array([5, 5, 20, 40, 40, 40, 40, 40, 40])),
        }),
    },
    'finger_explore': {
        'turn_easy_narrow': OrderedDict({
            'position': BoundedArray(name='position', shape=(4,),
                                     dtype=np.float32,
                                     minimum=np.array([-2, -2, -.1, -.1]),
                                     maximum=np.array([2, 2, .1, .1])),
            'velocity': BoundedArray(name='velocity', shape=(3,),
                                     dtype=np.float32,
                                     minimum=-12 * np.ones((3,)),
                                     maximum=12 * np.ones((3,))),
            'touch': BoundedArray(name='touch', shape=(2,),
                                     dtype=np.float32,
                                     minimum=np.zeros((2,)),
                                     maximum=5 * np.ones((2,))),
            'target_position': BoundedArray(name='target_position', shape=(2,),
                                     dtype=np.float32,
                                     minimum=np.array([-0.13, -0.13]),
                                     maximum=np.array([0.13, 0.13])),
            'dist_to_target': BoundedArray(name='dist_to_target', shape=(1,),
                                     dtype=np.float32,
                                     minimum=np.array([-0.2]),
                                     maximum=np.array([0.23])),
        }),
        'turn_hard_narrow': OrderedDict({
            'position': BoundedArray(name='position', shape=(4,),
                                     dtype=np.float32,
                                     minimum=np.array([-2, -2, -.1, -.1]),
                                     maximum=np.array([2, 2, .1, .1])),
            'velocity': BoundedArray(name='velocity', shape=(3,),
                                     dtype=np.float32,
                                     minimum=-12 * np.ones((3,)),
                                     maximum=12 * np.ones((3,))),
            'touch': BoundedArray(name='touch', shape=(2,),
                                     dtype=np.float32,
                                     minimum=np.zeros((2,)),
                                     maximum=5 * np.ones((2,))),
            'target_position': BoundedArray(name='target_position', shape=(2,),
                                     dtype=np.float32,
                                     minimum=np.array([-0.13, -0.13]),
                                     maximum=np.array([0.13, 0.13])),
            'dist_to_target': BoundedArray(name='dist_to_target', shape=(1,),
                                     dtype=np.float32,
                                     minimum=np.array([-0.2]),
                                     maximum=np.array([0.23])),
        }),
    },


}


def estimate_domains(env_name, task_name):
    import jax
    from dm_control import suite
    import utils
    from environments import point
    from environments import reacher_explore

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
        for _ in range(10):
            timestep = env.reset()
            for i in range(1000):
                observations.append(timestep.observation)
                a = policy()
                timestep = env.step(a)

    obs_stack = utils.tree_stack(observations)
    min_values = jax.tree_map(lambda x: x.min(axis=0), obs_stack)
    max_values = jax.tree_map(lambda x: x.max(axis=0), obs_stack)
    return min_values, max_values
