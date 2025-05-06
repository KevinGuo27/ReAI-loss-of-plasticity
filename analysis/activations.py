def register_layer_hooks(model, layer_names):
    activations = {}

    def get_hook(name):
        def hook(module, input, output):
            # Explicitly move tensor to CPU and make a copy
            activations[name] = output.detach().cpu()
        return hook

    # Debug: print all available layer names
    print("\nAvailable layers:")
    for name, _ in model.named_modules():
        print(f"  {name}")

    # Register hooks with error handling
    for name in layer_names:
        try:
            layer = dict(model.named_modules())[name]
            layer.register_forward_hook(get_hook(name))
        except KeyError:
            print(f"Warning: Layer {name} not found in model")
        except Exception as e:
            print(f"Error registering hook for {name}: {e}")

    return activations