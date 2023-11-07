# to create a new instance, there are two different way
#   1 = create a new insctance of the target class
#   2 = initialize the new instance with an appropriate initial state

# for first state = __new__ and __init__


class Point:
    def __new__(cls, *args, **kwargs):
        print("New instance of point")
        return super().__new__(cls)

    def __init__(self, x, y):
        print("New instance of point")
        self.x = x
        self.y = y

    def __repr__(self) -> str:
        return f"{type(self).__name__}(x={self.x}, y={self.y})"


def Runner7():
    point = Point(1, 3)
    print(point)


# Runner7()

class Dot:
    def __init__(self, a, b):
        print("Dot is initialized")
        self.a = a
        self.b = b


class Points:
    def __new__(cls, x, y):
        print("Initialize new Point of Dots")
        return Dot(x, y)

    def __init__(self, x, y):
        print("Initialize new Point of B")
        self.x = x
        self.y = y


def Runner():
    point = Points(1, 3)
    point.__init__(1, 3)


# Runner()


# inheritance

class Person:
    def __init__(self, name, birth_date):
        self.name = name
        self.birth_date = birth_date


class Employee(Person):
    def __init__(self, name, birth_date, position):
        super().__init__(name, birth_date)
        self.position = position


def Runner2():
    emp = Employee("t", "1.2.2", "gamer")

    print(emp.name)
    print(emp.position)
    print(emp.birth_date)


# Runner2()


# Returning instance of a different class

class Dog:
    def __init__(self):
        print("miaw")


class Cat:
    def __init__(self):
        print("miaw")


class Pet:
    def __new__(cls, animal):
        other = [Dog, Cat]
        instance = super().__new__(other[animal])
        print(f"I'm a {type(instance).__name__}!")

        return instance

    def __init__(self, animal):
        self.animal = animal
        print("Never runs!")


def Runner4():
    pet = Pet(0)


# Runner4()


# singleton pattern

class Singleton(object):
    instance = None

    def __new__(cls, *args, **kwargs):
        if cls.instance is None:
            cls.instance = super().__new__(cls)
        return cls.instance


def Runner5():
    s1 = Singleton()
    s2 = Singleton()
    print(s1 is s2)


# Runner5()



