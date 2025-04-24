def register_layer_hooks(model, layer_names):
    activations = {}

    def get_hook(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    for name in layer_names:
        layer = dict(model.named_modules())[name]
        layer.register_forward_hook(get_hook(name))

    return activations