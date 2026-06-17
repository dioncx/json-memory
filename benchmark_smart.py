import timeit

setup = """
import random

target_strings = ["warning", "mistake", "error", "problem", "issue", "avoid"]
random_strings = ["hello", "world", "test", "data", "foo", "bar", "baz", "qux"]

def generate_tags():
    res = []
    for _ in range(100):
        if random.random() < 0.1:
            res.append(random.choice(target_strings))
        else:
            res.append(random.choice(random_strings))
    return res

tags_list = [generate_tags() for _ in range(1000)]
"""

test_list = """
for tags in tags_list:
    any(tag in ["warning", "mistake", "error", "problem", "issue", "avoid"] for tag in tags)
"""

test_set = """
for tags in tags_list:
    any(tag in {"warning", "mistake", "error", "problem", "issue", "avoid"} for tag in tags)
"""

n = 1000
time_list = timeit.timeit(test_list, setup=setup, number=n)
time_set = timeit.timeit(test_set, setup=setup, number=n)

print(f"Time with list: {time_list:.6f} s")
print(f"Time with set:  {time_set:.6f} s")
print(f"Improvement:    {(time_list - time_set) / time_list * 100:.2f}%")
