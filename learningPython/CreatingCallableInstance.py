# we will use __call__
import time


# General implementation with functions
def printHello():
    print("hel1lo world")


def Runner():
    printHello.__call__()  # basic example with functions, it is same effect with directly calling


# Runner()


# General implementation with class
class Counter:
    def __init__(self):
        self.count = 0
        print("init is called")

    def increment(self):
        self.count += 1

    def printResult(self):
        print(self.count)

    def __call__(self, *args, **kwargs):
        self.increment()
        print("call is called")


def Runner2():
    counter = Counter()
    counter.increment()
    counter.printResult()


# General implementation with class 2

class Power:
    def __init__(self, exponent=2):
        self.exponent = exponent
        print("init is called")

    def __call__(self, base):
        print("call is called")
        return base ** self.exponent


def Runner3():
    power = Power(4)
    print(power(4))


# General implementation with stream clients
class Clients:
    def __init__(self):
        self.data = []
        self.sum: int = 0
        self.len: int = 0

    def result(self) -> float:
        return self.sum / self.len

    def __call__(self, val, **kwargs):
        self.data.append(val)
        self.sum += val
        self.len += 1


def Runner4():
    clientsStream = Clients()
    clientsStream(1)
    clientsStream(2)
    clientsStream(6)
    print(clientsStream.result())
    print(clientsStream.len)
    print(clientsStream.data)


# recursion with call and factorial with cache

class Factorial:
    def __init__(self):
        self.cache = {0: 1, 1: 1}

    def __call__(self, number):
        if number not in self.cache:
            self.cache[number] = number * self(number - 1)
        return self.cache[number]


def Runner5():
    fact = Factorial()
    print(fact(10))


# writing class based decorators

class ExecutionTimer:
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        start = time.perf_counter()
        result = self.func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{self.func.__name__}() tool {(end - start) * 1000:.4f} ms")
        return result


@ExecutionTimer
def square_number(numbers):
    lister = []
    for number in numbers:
        number = number ** 2
        lister.append(number)
    return lister


def Runner6():
    square_number(list(range(100)))


# Runner6()


# implementing the strategy design pattern



