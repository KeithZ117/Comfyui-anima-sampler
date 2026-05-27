"""ComfyUI custom-node entry point for the Anima sampler package."""

try:
    from .anima_sampler.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError:
    try:
        from anima_sampler.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    except Exception as exc:
        print(f"[comfyui-anima-sampler] Failed to load custom nodes: {exc}")
        NODE_CLASS_MAPPINGS = {}
        NODE_DISPLAY_NAME_MAPPINGS = {}
except Exception as exc:
    print(f"[comfyui-anima-sampler] Failed to load custom nodes: {exc}")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
