import asyncio
import random
import time
from server import PromptServer
from aiohttp import web

g_results = {}

if PromptServer.instance is not None:
    @PromptServer.instance.routes.post("/api/prompt_selector/response")
    async def handle_response(request):
        data = await request.json()
        node_id = int(data["unique_id"])
        selected = data.get("selected_prompt")
        g_results[node_id] = selected
        print(f"[PromptSelector] 收到选择：节点 {node_id} -> {selected}")
        return web.Response(text="ok")


class PromptSelectorWithTimeoutZV:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_list": ("STRING", {"multiline": True, "default": "一只猫\n一只狗\n一只鸟"}),
                "timeout_seconds": ("INT", {"default": 20, "min": 5, "max": 300, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff, "step": 1}),
            },
            "optional": {
                "default_selection": ("INT", {"default": 0, "min": 0, "step": 1}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID"
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("selected_prompt",)
    FUNCTION = "process_prompt"
    CATEGORY = "Prompt/Utils"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return random.random()

    @classmethod
    def IS_ASYNC(cls):
        return True

    async def process_prompt(self, prompt_list, timeout_seconds, default_selection, unique_id, seed):
        prompts = [line.strip() for line in prompt_list.split('\n') if line.strip()]
        if not prompts:
            return ("",)

        if default_selection is None:
            default_selection = 0
        default_index = max(0, min(default_selection, len(prompts) - 1))
        default_prompt = prompts[default_index]
        node_id = int(unique_id)

        g_results[node_id] = None

        # ---- 关键修改：将 WebSocket 发送任务提交到主事件循环 ----
        main_loop = PromptServer.instance.loop  # 主事件循环
        # 创建一个 coroutine 对象（注意：不能直接 await，需要在主循环中执行）
        send_coro = PromptServer.instance.send_json(
            "display-prompt-selector",
            {
                "unique_id": node_id,
                "options": prompts,
                "default_index": default_index,
                "timeout_seconds": timeout_seconds
            }
        )
        # 提交到主循环，并等待完成
        future = asyncio.run_coroutine_threadsafe(send_coro, main_loop)
        try:
            await asyncio.wrap_future(future)  # 等待发送完成
        except Exception as e:
            # 如果发送失败（比如 WebSocket 已断开），则使用默认值并退出
            print(f"[PromptSelector] 无法发送选择弹窗（WebSocket 错误）: {e}")
            del g_results[node_id]
            return (default_prompt,)

        print(f"[PromptSelector] 节点 {node_id} 开始等待用户选择...")
        start = time.time()
        while True:
            result = g_results.get(node_id)
            if result is not None:
                del g_results[node_id]
                return (result,)
            if time.time() - start >= timeout_seconds:
                del g_results[node_id]
                print(f"[PromptSelector] 节点 {node_id} 超时，使用默认值")
                return (default_prompt,)
            await asyncio.sleep(0.2)


NODE_CLASS_MAPPINGS = {
    "PromptSelectorWithTimeoutZV": PromptSelectorWithTimeoutZV,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptSelectorWithTimeoutZV": "Prompt Selector (with Timeout)",
}