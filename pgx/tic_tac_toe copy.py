# Copyright 2023 The Pgx Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
import jax.numpy as jnp

import pgx
from pgx._src.types import Array, PRNGKey


seed = 42
batch_size = 1
key = jax.random.PRNGKey(seed)


def act_randomly(rng_key, obs, mask):
    """Ignore observation and choose randomly from legal actions"""
    del obs
    probs = mask / mask.sum()
    logits = jnp.maximum(jnp.log(probs), jnp.finfo(probs.dtype).min)
    return jax.random.categorical(rng_key, logits=logits, axis=-1)


# Load the environment
env = pgx.make("tic_tac_toe")
init_fn = jax.jit(jax.vmap(env.init))
step_fn = jax.jit(jax.vmap(env.step))

# Initialize the states
key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, batch_size)
state = init_fn(keys)

# Run random simulation

ii = 0

while ii < 4:
#while not (state.terminated | state.truncated).all():
    key, subkey = jax.random.split(key)
    action = act_randomly(subkey, state.observation, state.legal_action_mask)
    state = step_fn(state, action)  # state.reward (2,)
    print(state)
    ii+=1