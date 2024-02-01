import copy
import os

import numpy as np

from geppo.common.initializers import init_seeds
from geppo.common.logger import Logger

class Evaluate:
    """
    Callback for evaluating an agent.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``

    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param callback_after_eval: Callback to trigger after every evaluation
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose: Verbosity level: 0 for no output, 1 for indicating information about evaluation results
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
        wrapped with a Monitor wrapper)
    """

    def __init__(
            self,
            model,
            runner,
            n_eval_episodes: int = 5,
            eval_freq: int = 10000,
            log_path: str = None,
            suffix: str = '',
            save_model: bool = False,
            deterministic: bool = True,
            device=None,
    ):
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.device = device

        self.save_model = save_model
        self.model = model
        self.best_model_save_path = log_path
        self.suffix = suffix

        self.runner = runner

        # Logs will be written in ``evaluations.npz``
        os.makedirs(name=log_path, exist_ok=True)
        if log_path is not None:
            if self.suffix != '':
                self.log_path = os.path.join(log_path, f"evaluations_{suffix}")
            else:
                self.log_path = os.path.join(log_path, f"evaluations")
        self.evaluations_returns = []
        self.evaluations_timesteps = []
        self.evaluations_successes = []
        # For computing success rate
        self._is_success_buffer = []

    def evaluate(self, t, train_env):
        # if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:

        self.eval_env = copy.deepcopy(train_env)
        returns, successes = self._evaluate()

        if self.log_path is not None:
            self.evaluations_timesteps.append(t)
            self.evaluations_returns.append(returns)
            self.evaluations_successes.append(successes)

            np.savez(
                self.log_path,
                timesteps=self.evaluations_timesteps,
                returns=self.evaluations_returns,
                successes=self.evaluations_successes,
            )

            mean_reward, std_reward = np.mean(returns), np.std(returns)
            mean_success, std_success = np.mean(successes), np.std(successes)

            self.last_mean_reward = mean_reward

            print(f"Eval num_timesteps={t}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
            print(f"Eval num_timesteps={t}, " f"episode_success={mean_success:.2f} +/- {std_success:.2f}")

            if mean_reward > self.best_mean_reward:
                print("New best mean reward!")
                self.best_mean_reward = mean_reward

        return mean_reward, std_reward

    def _evaluate(self):
        eval_returns = []
        eval_successes = []

        for episode_i in range(self.n_eval_episodes):
            ep_returns = []
            ep_successes = []
            J_tot = self.runner._generate_traj_eval(self.eval_env, self.model, deterministic=True)
            ep_returns.append(J_tot)
            ep_successes.append(False)

            eval_returns.append(J_tot)
            eval_successes.append(np.sum(ep_successes) * 100)

        return eval_returns, eval_successes


class BaseAlg:
    """Base algorithm class for training."""

    def __init__(self,seed,env,eval_env,actor,critic,runner,ac_kwargs,
        idx,save_dir,save_freq,checkpoint_file,keep_checkpoints):
        """Initializes BaseAlg class.

        Args:
            seed (int): random seed
            env (NormEnv): normalized environment
            actor (Actor): policy class
            critic (Critic): value function class
            runner (Runner): runner class to generate samples
            ac_kwargs (dict): dictionary containing actor and critic kwargs
            idx (int): index to associate with checkpoint files
            save_dir (str): path where checkpoint files are saved
            save_freq (float): number of steps between checkpoints
            checkpoint_file (str): name of checkpoint file for saving
            keep_checkpoints (bool): keep final dict of all checkpoints
        """

        self.seed = seed
        self.env = env
        self.actor = actor
        self.critic = critic
        self.runner = runner

        self.ac_kwargs = ac_kwargs

        self.save_dir = save_dir
        self.checkpoint_name = '%s_%d'%(checkpoint_file,idx)
        self.save_freq = save_freq
        self.keep_checkpoints = keep_checkpoints

        init_seeds(self.seed,env)
        self.logger = Logger()

        self.eval_module = Evaluate(self.actor, n_eval_episodes=20, eval_freq=10,
                                    log_path=save_dir, runner=self.runner)

    
    def _update(self):
        """Updates actor and critic."""
        raise NotImplementedError # Algorithm specific

    def learn(self,sim_size,no_op_batches,params):
        """Main training loop.

        Args:
            sim_size (float): number of steps to run training loop
            no_op_batches (int): number of no-op batches at start of training
                to initialize running normalization stats
            params (dict): dictionary of input parameters to pass to logger
        
        Returns:
            Name of checkpoint file.
        """

        checkpt_idx = 0
        if self.save_freq is None:
            checkpoints = np.array([sim_size])
        else:
            checkpoints = np.concatenate(
                (np.arange(0,sim_size,self.save_freq)[1:],[sim_size]))

        # No-op batches to initialize running normalization stats
        for _ in range(no_op_batches):
            self.runner.generate_batch(self.env,self.actor)
        s_raw, rtg_raw = self.runner.get_env_info()
        self.env.update_rms(s_raw,rtg_raw)
        self.runner.reset()

        # Main training loop
        sim_total = 0
        while sim_total < sim_size:
            self.runner.generate_batch(self.env,self.actor)

            self._update()

            s_raw, rtg_raw = self.runner.get_env_info()
            self.env.update_rms(s_raw,rtg_raw)

            log_info = self.runner.get_log_info()
            self.logger.log_train(log_info)

            print(sim_total, log_info)

            sim_total += self.runner.steps_total

            self.runner.update()

            # Save training data to checkpoint file
            if sim_total >= checkpoints[checkpt_idx]:
                # self.dump_and_save(params,sim_total)
                self.eval_module.evaluate(sim_total, train_env=self.env)
                checkpt_idx += 1

        return self.checkpoint_name
        
    def dump_and_save(self,params,steps):
        """Saves training data to checkpoint file and resets logger."""
        self.logger.log_params(params)

        final = {
            'actor_weights':    self.actor.get_weights(),
            'critic_weights':   self.critic.get_weights(),

            's_t':              self.env.s_rms.t_last,
            's_mean':           self.env.s_rms.mean,
            's_var':            self.env.s_rms.var,

            'r_t':              self.env.r_rms.t_last,
            'r_mean':           self.env.r_rms.mean,
            'r_var':            self.env.r_rms.var,

            'steps':            steps
        }
        self.logger.log_final(final)

        self.logger.dump_and_save(self.save_path,self.checkpoint_name,
            self.keep_checkpoints)
        self.logger.reset()