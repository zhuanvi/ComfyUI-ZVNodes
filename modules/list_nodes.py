from .utils import generate_node_mappings

import random

class RandomSelectListZV:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string_list": ("STRING", {"forceInput": True}),
                "n": ("INT", {"default": 1, "min": 1, "max": 99999, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("STRING",)
    FUNCTION = "process_list"
    CATEGORY = "ZVNodes/list"

    def process_list(self, string_list, n, seed):
        if isinstance(string_list, list):
            rng = random.Random(seed)
            if n > len(string_list):
                n = len(string_list)
            result_list = rng.sample(string_list, n)
        else:
            result_list = (string_list,)
        return (result_list, )

class JoinListZV:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string_list": ("STRING", {"forceInput": True}),
                "separator": ("STRING", {"default": ", "}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("STRING",)
    FUNCTION = "process"
    CATEGORY = "ZVNodes/list"

    def process(self, string_list, separator):
        if isinstance(string_list, list):
            return (separator.join(string_list),)
        else:
            return (string_list,)

NODE_CONFIG = {
    "RandomSelectListZV": {"class": RandomSelectListZV, "name": "Random Select From List"},
    "JoinListZV": {"class": JoinListZV, "name": "Join List"}
}

NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = generate_node_mappings(NODE_CONFIG)