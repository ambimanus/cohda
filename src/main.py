# coding=utf-8

import scenarios
import cli


if __name__ == '__main__':
    sc = scenarios.SC(
        sim_msg_delay_min=None,
        sim_msg_delay_max=None,
        sim_agent_delay_min=None,
        sim_agent_delay_max=None,
    )

    cli.run(sc)
