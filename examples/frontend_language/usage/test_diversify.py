"""
Usage:
python3 local_example_complete.py
"""

import sglang as sgl


@sgl.function
def few_shot_qa(s, question):
    s += """The following are questions with answers.
Q: What is the capital of France?
A: Paris
Q: What is the capital of Germany?
A: Berlin
Q: What is the capital of Italy?
A: Rome
"""
    s += "Q: " + question + "\n"
    s += "A:" + sgl.gen("answer", stop="\n", temperature=0)


def single():
    state = few_shot_qa.run(question="What is the capital of the United States?")
    answer = state["answer"].strip().lower()


    print(state.text())


def stream():
    state = few_shot_qa.run(
        question="What is the capital of the United States?", stream=True
    )

    for out in state.text_iter("answer"):
        print(out, end="", flush=True)
    print()


def batch():
    states = few_shot_qa.run_batch(
        [
            {"question": "What is the capital of the United States?"},
            {"question": "What is the capital of the United States?"},
            {"question": "What is the capital of the United States?"},
            {"question": "What is the capital of the United States?"},
            {"question": "What is the capital of the United States?"},
            {"question": "What is the capital of China?"},
            {"question": "What is the capital of China?"},
            {"question": "What is the capital of China?"},
            {"question": "What is the capital of China?"},
            {"question": "What is the capital of China?"},
            {"question": "What is the capital of China?"},
            {"question": "What is the capital of Chopped Chin?"},
        ]
    )

    for s in states:
        print(s["answer"])


if __name__ == "__main__":
    runtime = sgl.Runtime(model_path="meta-llama/Llama-3.2-1B-Instruct", mem_fraction_static = 0.7)
    sgl.set_default_backend(runtime)

    batch()