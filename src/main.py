# coding=utf-8

import scenarios
import cli
from configuration import Configuration


if __name__ == '__main__':
    cfg = Configuration(
        seed=0,
        msg_delay_min=None,
        msg_delay_max=None,
        agent_delay_min=None,
        agent_delay_max=None,
        log_to_file=False,
    )
    sc = scenarios.SC(cfg.rnd, cfg.seed, opt_h=5)
    # sc = scenarios.SVSM(cfg.rnd, cfg.seed)
    # sc = scenarios.CHP(cfg.rnd, cfg.seed, opt_m=10, opt_n=400, opt_q=16, opt_q_constant=20.0)
    #
    # sc = scenarios.SC(cfg.rnd, cfg.seed, opt_h=5,
    #                   agent_type='AgentStigspaceMMMSSP')
    # sc = scenarios.SVSM(cfg.rnd, cfg.seed, agent_type='AgentStigspaceSVSM')
    # sc = scenarios.CHP(cfg.rnd, cfg.seed, opt_m=30, opt_n=200,
    #                    agent_type='AgentStigspaceMMMSSP')
    cfg.scenario = sc

    cli.run(cfg)
