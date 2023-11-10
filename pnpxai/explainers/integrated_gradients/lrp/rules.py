from captum.attr._utils.lrp_rules import EpsilonRule, GammaRule

def epsilon_rule_factory(epsilon=1e-9):
    class CustomEpsilonRule(EpsilonRule):
        def __init__(self):
            super().__init__(epsilon=epsilon)
    return CustomEpsilonRule


def gamma_rule_factory(gamma=0.25, set_bias_to_zero=False):
    class CustomGammaRule(GammaRule):
        def __init__(self):
            super().__init__(
                gamma = gamma,
                set_bias_to_zero = set_bias_to_zero
            )
    return CustomGammaRule
