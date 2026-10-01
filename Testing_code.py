def hardlims(n):
    return -1 if n < 0 else 1

def neuron(p, w=1, b=-3):
    n = w * p + b
    return hardlims(n)

def neuron2(p, w=2, b=-6):
    n = w * p + b
    return hardlims(n)

# Test around the switching point p = 3
for p in [1, 2, 2.9, 3, 3.1, 4, 5]:
    print(f"p = {p:>4} -> n = {p*1 - 3:>5} -> a = {neuron(p)}")
    print(f"p = {p:>4} -> n = {p*1 - 3:>5} -> a = {neuron2(p)}")

### Switching point is always p = (-b/w)
