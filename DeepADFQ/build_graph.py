"""Deep ADFQ learning graph

This code was written on top of OpenAI baseline code - baselines/baselines/deepq/build_graph.py
Therefore, most parts of this code and its structure are same with the original code.
The major difference is the output size of the network (2*num_actions) and build_train(), 
build_act_greedy(), and build_act_bayesian()

The functions in this file can are used to create the following functions:
======= act ========
    Function to chose an action given an observation
    Parameters
    ----------
    observation: object
        Observation that can be feed into the output of make_obs_ph
    stochastic: bool
        if set to False all the actions are always deterministic (default False)
    update_eps_ph: float
        update epsilon a new value, if negative not update happens
        (default: no update)
    Returns
    -------
    Tensor of dtype tf.int64 and shape (BATCH_SIZE,) with an action to be performed for
    every element of the batch.
======= act (in case of parameter noise) ========
    Function to chose an action given an observation
    Parameters
    ----------
    observation: object
        Observation that can be feed into the output of make_obs_ph
    stochastic: bool
        if set to False all the actions are always deterministic (default False)
    update_eps_ph: float
        update epsilon a new value, if negative not update happens
        (default: no update)
    reset_ph: bool
        reset the perturbed policy by sampling a new perturbation
    update_param_noise_threshold_ph: float
        the desired threshold for the difference between non-perturbed and perturbed policy
    update_param_noise_scale_ph: bool
        whether or not to update the scale of the noise for the next time it is re-perturbed
    Returns
    -------
    Tensor of dtype tf.int64 and shape (BATCH_SIZE,) with an action to be performed for
    every element of the batch.
======= train =======
    Function that takes a transition (s,a,r,s') and optimizes Bellman equation's error:
        td_error = Q(s,a) - (r + gamma * max_a' Q(s', a'))
        loss = huber_loss[td_error]
    Parameters
    ----------
    obs_t: object
        a batch of observations
    action: np.array
        actions that were selected upon seeing obs_t.
        dtype must be int32 and shape must be (batch_size,)
    reward: np.array
        immediate reward attained after executing those actions
        dtype must be float32 and shape must be (batch_size,)
    obs_tp1: object
        observations that followed obs_t
    done: np.array
        1 if obs_t was the last observation in the episode and 0 otherwise
        obs_tp1 gets ignored, but must be of the valid shape.
        dtype must be float32 and shape must be (batch_size,)
    weight: np.array
        imporance weights for every element of the batch (gradient is multiplied
        by the importance weight) dtype must be float32 and shape must be (batch_size,)
    Returns
    -------
    td_error: np.array
        a list of differences between Q(s,a) and the target in Bellman's equation.
        dtype is float32 and shape is (batch_size,)
======= update_target ========
    copy the parameters from optimized Q function to the target Q function.
    In Q learning we actually optimize the following error:
        Q(s,a) - (r + gamma * max_a' Q'(s', a'))
    Where Q' is lagging behind Q to stablize the learning. For example for Atari
    Q' is set to Q once every 10000 updates training steps.
"""
import tensorflow as tf
import baselines0.common.tf_util as U
import numpy as np


def scope_vars(scope, trainable_only=False):
    """
    Get variables inside a scope
    The scope can be specified as a string
    Parameters
    ----------
    scope: str or VariableScope
        scope in which the variables reside.
    trainable_only: bool
        whether or not to return only the variables that were marked as trainable.
    Returns
    -------
    vars: [tf.Variable]
        list of variables in `scope`.
    """
    return tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES if trainable_only else tf.GraphKeys.GLOBAL_VARIABLES,
        scope=scope if isinstance(scope, str) else scope.name
    )


def scope_name():
    """Returns the name of current scope as a string, e.g. deepadfq/q_func"""
    return tf.get_variable_scope().name


def absolute_scope_name(relative_scope_name):
    """Appends parent scope name to `relative_scope_name`"""
    return scope_name() + "/" + relative_scope_name


def build_act(make_obs_ph, q_func, num_actions, scope="deepadfq", reuse=None):
    """Creates the act function:
    Parameters
    ----------
    make_obs_ph: str -> tf.placeholder or TfInput
        a function that take a name and creates a placeholder of input with that name
    q_func: (tf.Variable, int, str, bool) -> tf.Variable
        the model that takes the following inputs:
            observation_in: object
                the output of observation placeholder
            num_actions: int
                number of actions
            scope: str
            reuse: bool
                should be passed to outer variable scope
        and returns a tensor of shape (batch_size, num_actions) with values of every action.
    num_actions: int
        number of actions.
    scope: str or VariableScope
        optional scope for variable_scope.
    reuse: bool or None
        whether or not the variables should be reused. To be able to reuse the scope must be given.
    Returns
    -------
    act: (tf.Variable, bool, float) -> tf.Variable
        function to select and action given observation.
`       See the top of the file for details.
    """
    with tf.variable_scope(scope, reuse=reuse):
        observations_ph = make_obs_ph("observation")
        stochastic_ph = tf.placeholder(tf.bool, (), name="stochastic")
        update_eps_ph = tf.placeholder(tf.float32, (), name="update_eps")

        eps = tf.get_variable("eps", (), initializer=tf.constant_initializer(0), dtype=tf.float32)

        # batch_size by action* 2 matrix. The first action_dim number of columns is the mean. 
        # The last action_dim number of columns is the variance
        q_values = q_func(observations_ph.get(), num_actions*2, scope="q_func")
        deterministic_actions = tf.argmax(q_values[:, :num_actions], axis=1)

        batch_size = tf.shape(observations_ph.get())[0]

        random_actions = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=num_actions, dtype=tf.int64)
        chose_random = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < eps
        stochastic_actions = tf.where(chose_random, random_actions, deterministic_actions)

        output_actions = tf.cond(stochastic_ph, lambda: stochastic_actions, lambda: deterministic_actions)
        update_eps_expr = eps.assign(tf.cond(update_eps_ph >= 0, lambda: update_eps_ph, lambda: eps))
        _act = U.function(inputs=[observations_ph, stochastic_ph, update_eps_ph],
                         outputs=output_actions,
                         givens={update_eps_ph: -1.0, stochastic_ph: True},
                         updates=[update_eps_expr])
        def act(ob, stochastic=True, update_eps=-1):
            return _act(ob, stochastic, update_eps)
        return act

def build_act_greedy(make_obs_ph, q_func, num_actions, scope="deepadfq", reuse=True, eps=0.0):
    """Creates the act function for a simple fixed epsilon greedy
    """
    with tf.variable_scope(scope, reuse=reuse):
        observations_ph = make_obs_ph("observation")
        stochastic_ph = tf.placeholder(tf.bool, (), name="stochastic")

        q_values = q_func(observations_ph.get(), num_actions*2, reuse=tf.AUTO_REUSE, scope="q_func")
        deterministic_actions = tf.argmax(q_values[:, :num_actions], axis=1)

        batch_size = tf.shape(observations_ph.get())[0]

        random_actions = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=num_actions, dtype=tf.int64)
        chose_random = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < eps
        stochastic_actions = tf.where(chose_random, random_actions, deterministic_actions)

        output_actions = tf.cond(stochastic_ph, lambda: stochastic_actions, lambda: deterministic_actions)
        _act = U.function(inputs=[observations_ph, stochastic_ph],
                         outputs=output_actions)
        def act(ob, stochastic=True):
            return _act(ob, stochastic)
        return act

def build_act_bayesian(make_obs_ph, q_func, num_actions, scope="deepadfq", reuse=None):
    """Creates the act function for Bayesian sampling
    """
    with tf.variable_scope(scope, reuse=reuse):
        observations_ph = make_obs_ph("observation")
        q_values = q_func(observations_ph.get(), num_actions*2, reuse=tf.AUTO_REUSE, scope="q_func") # mean and -log(sd)
        q_means = q_values[:,:num_actions]
        q_sds = tf.exp(-q_values[:,num_actions:])
        samples = tf.random_normal((),mean=q_means,stddev=q_sds)
        output_actions = tf.argmax(samples, axis=1)

        _act = U.function(inputs=[observations_ph],
                         outputs=output_actions
                         )
        def act(ob, stochastic=True, update_eps=-1):
            return _act(ob)
        return act

def build_train(sess, make_obs_ph, q_func, num_actions, optimizer, grad_norm_clipping=None, gamma=0.9,
    scope="deepadfq", reuse=None, varTH=1e-05, test_eps=0.05, act_policy='egreedy'):
    """Creates the train function:
    Parameters
    ----------
    make_obs_ph: str -> tf.placeholder or TfInput
        a function that takes a name and creates a placeholder of input with that name
    q_func: (tf.Variable, int, str, bool) -> tf.Variable
        the model that takes the following inputs:
            observation_in: object
                the output of observation placeholder
            num_actions: int
                number of actions
            scope: str
            reuse: bool
                should be passed to outer variable scope
        and returns a tensor of shape (batch_size, num_actions) with values of every action.
    num_actions: int
        number of actions
    reuse: bool
        whether or not to reuse the graph variables
    optimizer: tf.train.Optimizer
        optimizer to use for the Q-learning objective.
    grad_norm_clipping: float or None
        clip gradient norms to this value. If None no clipping is performed.
    gamma: float
        discount rate.
    scope: str or VariableScope
        optional scope for variable_scope.
    reuse: bool or None
        whether or not the variables should be reused. To be able to reuse the scope must be given.
    param_noise: bool
        whether or not to use parameter space noise (https://arxiv.org/abs/1706.01905)
    param_noise_filter_func: tf.Variable -> bool
        function that decides whether or not a variable should be perturbed. Only applicable
        if param_noise is True. If set to None, default_param_noise_filter is used by default.
    varTH : float
        variance threshold
    test_eps : float
        epsilon value for epsilon-greedy method in evaluation
    act_policy : str
        either 'egreedy' or 'bayesian' for action policy
        
    Returns
    -------
    act: (tf.Variable, bool, float) -> tf.Variable
        function to select and action given observation.
`       See the top of the file for details.
    train: (object, np.array, np.array, object, np.array, np.array) -> np.array
        optimize the error in Bellman's equation.
`       See the top of the file for details.
    update_target: () -> ()
        copy the parameters from optimized Q function to the target Q function.
`       See the top of the file for details.
    debug: {str: function}
        a bunch of functions to print debug data like q_values.
    """
    if act_policy == 'egreedy':
        act_f = build_act(make_obs_ph, q_func, num_actions, scope=scope, reuse=reuse)
    elif act_policy == 'bayesian':
        act_f = build_act_bayesian(make_obs_ph, q_func, num_actions, scope=scope, reuse=reuse)
    else:
        raise ValueError("Please choose either egreedy or bayesian for action policy.")
    act_greedy = build_act_greedy(make_obs_ph, q_func, num_actions, scope=scope, reuse=True, eps=test_eps)

    sdTH = np.sqrt(varTH, dtype = np.float32)

    with tf.variable_scope(scope, reuse=reuse):
        # set up placeholders
        obs_t_input = make_obs_ph("obs_t")
        act_t_ph = tf.placeholder(tf.int32, [None], name="action")
        rew_t_ph = tf.placeholder(tf.float32, [None], name="reward")
        obs_tp1_input = make_obs_ph("obs_tp1")
        done_mask_ph = tf.placeholder(tf.float32, [None], name="done")
        importance_weights_ph = tf.placeholder(tf.float32, [None], name="weight")


        target_means = tf.placeholder(tf.float32, [None], name="target_means")
        target_sd = tf.placeholder(tf.float32, [None], name="target_sd")

        # q network evaluation
        q_t = q_func(obs_t_input.get(), num_actions*2, scope="q_func", reuse=True)  # reuse parameters from act
        q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name + "/q_func")

        # target q network evalution
        q_tp1 = q_func(obs_tp1_input.get(), num_actions*2, scope="target_q_func")
        
        target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name + "/target_q_func")

        mean_values = q_t[:, :num_actions]
        rho_values = q_t[:, num_actions:]

        mean_selected = tf.reduce_sum(mean_values * tf.one_hot(act_t_ph, num_actions, dtype=tf.float32), 1)
        rho_selected = tf.reduce_sum(rho_values * tf.one_hot(act_t_ph, num_actions, dtype=tf.float32), 1)

        sd_selected = tf.exp(-rho_selected)

        mean_error = target_means - mean_selected
        sd_error = target_sd - sd_selected 
        huber_loss = U.huber_loss(mean_error) + U.huber_loss(sd_error)
        weighted_loss = tf.reduce_mean(huber_loss * importance_weights_ph)
        
        #kl_loss = tf.contrib.distributions.kl_divergence(
        #    tf.distributions.Normal(loc=target_means, scale=target_sd),
        #    tf.distributions.Normal(loc=mean_selected, scale=sd_selected),
        #    name='kl_loss')
        #weighted_loss = tf.reduce_mean(kl_loss * importance_weights_ph)
        if grad_norm_clipping is not None:
            gradients = optimizer.compute_gradients(weighted_loss, var_list=q_func_vars)
            for i, (grad, var) in enumerate(gradients):
                if grad is not None:
                    gradients[i] = (tf.clip_by_norm(grad, grad_norm_clipping), var)
            optimize_expr = optimizer.apply_gradients(gradients)
        else:
            optimize_expr = optimizer.minimize(weighted_loss, var_list=q_func_vars)


        update_target_expr = []
        for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name),
                                   sorted(target_q_func_vars, key=lambda v: v.name)):
            update_target_expr.append(var_target.assign(var))
        update_target_expr = tf.group(*update_target_expr)

        # Create callable functions
        train = U.function(
            inputs=[
                obs_t_input,
                act_t_ph,
                target_means,
                target_sd,
                importance_weights_ph
            ],
            outputs=[huber_loss, tf.reduce_mean(tf.abs(mean_error)), tf.reduce_mean(tf.abs(sd_error))],
            updates=[optimize_expr]
        )
        update_target = U.function([], [], updates=[update_target_expr])

        q_target_vals = U.function(
            inputs=[obs_tp1_input],
            outputs = [q_tp1])

        return act_f, act_greedy, q_target_vals, train, update_target
