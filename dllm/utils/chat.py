import shutil
import textwrap
from typing import List, Literal

import dllm

# ============================================================
# Utility helpers
# ============================================================

try:
    L = shutil.get_terminal_size().columns
    if not isinstance(L, int) or L <= 0:
        L = 120
except Exception:
    L = 120
DIV = "=" * L
SUB = "-" * L


def banner_line(text: str, width: int = L, fill: str = "=") -> str:
    """Return a centered banner line with given width and fill."""
    text = f" {text.strip()} "
    fill_len = width - len(text)
    if fill_len <= 0:
        return text
    left = fill_len // 2
    right = fill_len - left
    return f"{fill * left}{text}{fill * right}"


def print_wrapped(text: str, width: int = L):
    """Print text with automatic line wrapping."""
    wrapped = textwrap.fill(text, width=width)
    print(wrapped)


def boxed(text: str, width: int = L, padding: int = 1):
    """Render a centered box with the given text and width."""
    lines = text.splitlines()
    content_width = max(len(line) for line in lines)
    box_width = min(width, content_width + padding * 2 + 2)

    # compute left margin for centering
    terminal_width = width
    left_margin = max((terminal_width - box_width) // 2, 0)
    margin = " " * left_margin

    top = margin + "┌" + "─" * (box_width - 2) + "┐"
    bottom = margin + "└" + "─" * (box_width - 2) + "┘"

    print(top)
    for line in lines:
        inner = line.center(content_width)
        print(margin + "│" + " " * padding + inner + " " * padding + "│")
    print(bottom)


def render_menu(round_idx: int):
    """Render a boxed menu of possible actions."""
    if round_idx == 0:
        text = (
            "Possible next actions:\n"
            "[1] Continue this chat\n"
            "[2] End this chat and start a new one\n"
            "[3] Exit"
        )
    else:
        text = (
            f"(Round {round_idx})\n"
            "Possible next actions:\n"
            "[1] Continue this chat\n"
            "[2] End this chat and start a new one\n"
            "[3] Exit"
        )

    print()  # spacing
    boxed(text)


def prompt_choice() -> Literal["1", "2", "3"]:
    while True:
        print("Select action [1/2/3]: ")
        choice = input().strip()
        if choice in ("1", "2", "3"):
            return choice
        print(banner_line("<Invalid choice. Please type 1, 2, or 3.>", fill=" "))


def build_chat_inputs(tokenizer, messages: List[dict], add_generation_prompt: bool):
    """Tokenize chat messages into inputs tensor."""
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        tokenize=True,
    )


def visualize_histories(tokenizer, histories):
    try:
        terminal_visualizer = dllm.utils.TerminalVisualizer(tokenizer=tokenizer)
        terminal_visualizer.visualize(histories, rich=True)
    except Exception as e:
        print(f"(Visualization skipped: {e})")


# ============================================================
# Modes
# ============================================================
def single_turn_sampling(sampler, sampler_config, visualize: bool):
    print()
    print(banner_line("continuation mode"))
    model, tokenizer = sampler.model, sampler.tokenizer

    while True:
        print(banner_line("<Type your prompt below. Press Ctrl+C to exit.>", fill=" "))
        try:
            # user_text = input("Prompt > ").strip()
            print("[Prompt] > ")
            user_text = input().strip()
        except (EOFError, KeyboardInterrupt):
            print("\n" + banner_line("Exiting. Bye!", width=len(DIV)))
            return

        # if not user_text:
        #     print("(Empty input, skipped)\n")
        #     continue

        inputs = tokenizer([user_text], add_special_tokens=False)["input_ids"]
        outputs = sampler.sample(inputs, sampler_config, return_dict=True)
        text = dllm.utils.decode_trim(tokenizer, outputs.sequences.tolist(), inputs)[0]

        print(banner_line("Output"))
        print_wrapped(text if text else "<empty>")
        print(DIV + "\n")

        if visualize:
            visualize_histories(tokenizer, outputs.histories)


def multi_turn_chat(sampler, sampler_config, visualize: bool):
    # """Chat mode with chat template & message history."""
    print()
    print(banner_line("multi-turn chat mode"))
    print(banner_line("<Starting a new chat. Type your message.>", fill=" "))
    model, tokenizer = sampler.model, sampler.tokenizer

    messages: List[dict] = []
    round_idx = 0

    while True:
        try:
            print("[You]:")
            user_msg = input().strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting. Bye!")
            return

        messages.append({"role": "user", "content": user_msg})
        inputs = build_chat_inputs(tokenizer, [messages], add_generation_prompt=True)

        outputs = sampler.sample(inputs, sampler_config, return_dict=True)
        reply = dllm.utils.decode_trim(tokenizer, outputs.sequences.tolist(), inputs)[0]

        print(DIV)
        print_wrapped("[Assistant]: " + reply if reply else "<empty>")
        print(DIV + "\n")

        messages.append({"role": "assistant", "content": reply})

        if visualize:
            visualize_histories(tokenizer, outputs.histories)

        render_menu(round_idx)
        choice = prompt_choice()
        if choice == "1":
            print(banner_line("<Type your message.>", fill=" "))
            round_idx += 1
            continue
        elif choice == "2":
            print(banner_line("<Starting a new chat. Type your message.>", fill=" "))
            messages = []
            round_idx = 0
            continue
        else:
            print("\nExiting. Bye!")
            return
