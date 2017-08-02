import argparse
import gym
from gym import wrappers
import os.path as osp
import random
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

import dqn
from dqn_utils import *
from atari_wrappers import *


def cartpole_model(img_in, num_actions, scope, reuse=False):
    # as described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    with tf.variable_scope(scope, reuse=reuse):
        # out = tf.ones(tf.shape(img_in))
        out = img_in
        out = layers.flatten(out)
        with tf.variable_scope("action_value"):
            out = layers.fully_connected(out, num_outputs=16,
                    activation_fn=tf.nn.relu, scope='fc_input')
            out = layers.fully_connected(out, num_outputs=num_actions,
                    activation_fn=None, scope='fc_head')

        return out

def cartpole_learn(env, session, num_timesteps):
    # This is just a rough estimate
    num_iterations = float(num_timesteps) / 4.0

    # lr_multiplier = 1.0
    # lr_multiplier = 0.1
    # lr_schedule = PiecewiseSchedule([
                                         # (0,                   1e-4 * lr_multiplier),
                                         # (num_iterations / 2,  1e-5 * lr_multiplier),
                                    # ],
                                    # outside_value=5e-5 * lr_multiplier)
    lr_schedule = InverseSchedule(initial_p=0.1, gamma=0.6)

    optimizer = dqn.OptimizerSpec(
        constructor=tf.train.GradientDescentOptimizer,
        # constructor=tf.train.AdamOptimizer,
        # kwargs=dict(epsilon=1e-4),
        kwargs=dict(),
        # constructor=tf.train.RMSPropOptimizer,
        # kwargs=dict(epsilon=1e-1),
        lr_schedule=lr_schedule
    )

    def stopping_criterion(env, t):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps

    exploration_schedule = PiecewiseSchedule(
        [
            (0, 1.0),
            # (0.2 * num_timesteps, 0.9),
            # (0.5 * num_timesteps, 0.5),
            (0.1 * num_timesteps, 0.1),
        ], outside_value=0.01
    )

    dqn.learn(
        env,
        q_func=cartpole_model,
        optimizer_spec=optimizer,
        session=session,
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion,
        replay_buffer_size=100000,
        batch_size=256,
        gamma=0.99,
        learning_starts=2000,
        learning_freq=1,
        frame_history_len=4,
        target_update_freq=1000,
        grad_norm_clipping=1000,
    )
    env.close()

def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.physical_device_desc for x in local_device_protos if x.device_type == 'GPU']

def set_global_seeds(i):
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(i) 
    np.random.seed(i)
    random.seed(i)

def get_session():
    tf.reset_default_graph()
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    session = tf.Session(config=tf_config)
    print("AVAILABLE GPUS: ", get_available_gpus())
    return session

def get_env(task, seed):
    env_id = task.env_id

    env = gym.make(env_id)

    set_global_seeds(seed)
    env.seed(seed)

    expt_dir = '/tmp/hw3_vid_dir2/'
    env = wrappers.Monitor(env, osp.join(expt_dir, "gym"), force=True)
    env = wrap_deepmind(env)

    return env

def main():
    # Run training
    max_timesteps = 100000
    seed = 0 # Use a seed of zero (you may want to randomize the seed!)
    env = gym.make("CartPole-v0")
    env.seed(seed)
    set_global_seeds(seed)
    env = wrappers.Monitor(env, '/tmp/cartpole-experiment-1', force=True)
    session = get_session()
    cartpole_learn(env, session, num_timesteps=max_timesteps)

if __name__ == "__main__":
    main()
