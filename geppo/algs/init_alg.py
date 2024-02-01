"""Interface to all algorithm files."""
from geppo.algs.geppo import GePPO

gen_algs = ['geppo']

def init_alg(sim_seed,env,eval_env,actor,critic,runner,ac_kwargs,alg_name,
    idx,save_dir,save_freq,checkpoint_file,keep_checkpoints):
    """Initializes algorithm."""

    if alg_name in ['ppo','geppo']:    
        alg = GePPO(sim_seed,env,eval_env,actor,critic,runner,ac_kwargs,
            idx,save_dir,save_freq,checkpoint_file,keep_checkpoints)
    else:
        raise ValueError('invalid alg_name')

    return alg