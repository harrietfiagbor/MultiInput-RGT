class HookActivation:
    def __init__(self, target_layer):
        """Initialize a Pytorch hook using `hook_activation` function."""

        self.hook = target_layer.register_forward_hook(self.hook_activation)

    def hook_activation(self, target_layer, activ_in, activ_out):
        """Create a copy of the layer output activations and save
        in `self.stored`.
        """
        self.stored = activ_out.detach().clone()

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.hook.remove()


class HookGradient:
    def __init__(self, target_layer):
        """Initialize a Pytorch hook using `hook_gradient` function."""

        self.hook = target_layer.register_backward_hook(self.hook_gradient)

    def hook_gradient(self, target_layer, gradient_in, gradient_out):
        """Create a copy of the layer output gradients and save
        in `self.stored`.
        """
        self.stored = gradient_out[0].detach().clone()

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.hook.remove()
