import numpy as np

from .scm import sanity_3_lin,sanity_3_non_add,sanity_3_non_add_output,sanity_3_lin_output,sanity_3_non_lin_output, sanity_3_non_lin,economic_growth_china,economic_growth_china_output, nutrition_model, nutrition_model_output,german_credit,german_credit_output
scm_dict = {
    "sanity-3-lin": sanity_3_lin,
    "sanity-3-lin-output": sanity_3_lin_output,
    "sanity-3-non-lin": sanity_3_non_lin,
    "sanity-3-non-lin-output": sanity_3_non_lin_output,
    "sanity-3-non-add": sanity_3_non_add,
    "sanity-3-non-add-output": sanity_3_non_add_output,
    #"adult":adult_model, 
    "credit-output":german_credit_output,
    "nutrition": nutrition_model,
    "nutrition-output":nutrition_model_output,
    "credit":german_credit,
    'economic-output':economic_growth_china_output,
    'economic':economic_growth_china,

}


def _remove_prefix(node):
    """replaces e.g. x101 or u101 with 101"""
    assert node[0] == "x" or node[0] == "u"
    return node[1:]


def load_scm_equations(scm_class: str):
    ###########################
    #  loading scm equations  #
    ###########################
    (
        structural_equations_np,
        structural_equations_ts,
        noise_distributions,
        continuous,
        categorical,
        immutables,
    ) = scm_dict[scm_class]()

    ###########################
    #       some checks       #
    ###########################
    # TODO duplicate with tests
    if not (
        [_remove_prefix(node) for node in structural_equations_np.keys()]
        == [_remove_prefix(node) for node in structural_equations_ts.keys()]
        == [_remove_prefix(node) for node in noise_distributions.keys()]
    ):  
        raise ValueError(
            "structural_equations_np & structural_equations_ts & noises_distributions should have identical keys."
        )

    if not (
        np.all(["x" in node for node in structural_equations_np.keys()])
        and np.all(["x" in node for node in structural_equations_ts.keys()])
    ):
        raise ValueError("endogenous variables must start with `x`.")

    return (
        structural_equations_np,
        structural_equations_ts,
        noise_distributions,
        continuous,
        categorical,
        immutables,
    )
