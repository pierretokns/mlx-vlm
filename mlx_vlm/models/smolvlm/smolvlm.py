from ..idefics3 import Model as Idefics3Model


class Model(Idefics3Model):
    """SmolVLM model that inherits from Idefics3.

    Uses the parent's _prepare_inputs_for_multimodal which handles
    variable numbers of image tokens via masked_scatter approach.
    """
    pass
