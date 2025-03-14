# region: tuple functions

# add tuples, return new tuple
def add(
    lhs: tuple[int | float, ...] | int | float,
    rhs: tuple[int | float, ...] | int | float,
):
    if isinstance(lhs, tuple) and isinstance(rhs, tuple):
        return tuple([x + y for x, y in zip(lhs, rhs)])
    elif isinstance(lhs, tuple):
        return tuple([x + rhs for x in lhs])  # type:ignore
    elif isinstance(rhs, tuple):
        return tuple([x + lhs for x in rhs])
    else:
        return lhs + rhs


# sub tuples, return new tuple
def sub(
    a: tuple[int | float, ...] | int | float, b: tuple[int | float, ...] | int | float
) -> tuple[int | float, ...] | int | float:
    if isinstance(a, tuple) and isinstance(b, tuple):
        return tuple([x - y for x, y in zip(a, b)])
    elif isinstance(a, tuple):
        return tuple([x - b for x in a])  # type:ignore
    elif isinstance(b, tuple):
        return tuple([x - a for x in b])
    else:
        return a - b


# multiply tuples element-wise, return new tuple
def mult(
    a: tuple[int | float, ...] | int | float, b: tuple[int | float, ...] | int | float
) -> tuple[int | float, ...] | int | float:
    if isinstance(a, tuple) and isinstance(b, tuple):
        return tuple([x * y for x, y in zip(a, b)])
    elif isinstance(a, tuple):
        return tuple([x * b for x in a])  # type:ignore
    elif isinstance(b, tuple):
        return tuple([x * a for x in b])
    else:
        return a * b


# divide tuples element-wise, return new tuple
def div(
    a: tuple[int | float, ...] | int | float, b: tuple[int | float, ...] | int | float
) -> tuple[int | float, ...] | int | float:
    if isinstance(a, tuple) and isinstance(b, tuple):
        return tuple([x / y for x, y in zip(a, b)])
    elif isinstance(a, tuple):
        return tuple([x / b for x in a])  # type:ignore
    elif isinstance(b, tuple):
        return tuple([x / a for x in b])
    else:
        return a / b


# convert float | int tuple to int tuple
def round_tuple(t: tuple[int | float, ...]) -> tuple[int, ...]:
    return tuple([round(x) for x in t])


# apply abs to all elements of tuple
def abs_tuple(t: tuple[int | float, ...]) -> tuple[int | float, ...]:
    return tuple([abs(x) for x in t])


# negate all elements of tuple (*-1)
def neg_tuple(t: tuple[int | float, ...]) -> tuple[int | float, ...]:
    return tuple([-x for x in t])


# return true if each element in a is less than b
def LT_tuple(a: tuple[int | float, ...], b: tuple[int | float, ...]) -> bool:
    return all([x < y for x, y in zip(a, b)])


# return true if each element in a is greater than b
def GT_tuple(a: tuple[int | float, ...], b: tuple[int | float, ...]) -> bool:
    return all([x > y for x, y in zip(a, b)])


# return true if each element in a is equal to b
def EQ_tuple(a: tuple[int | float, ...], b: tuple[int | float, ...]) -> bool:
    return all([x == y for x, y in zip(a, b)])


# return true if each element in a <= b <= c
def BTW_tuple(
    a: tuple[int | float, ...] | int | float,
    b: tuple[int | float, ...] | int | float,
    c: tuple[int | float, ...] | int | float,
) -> bool:
    # first check if all inputs are non-tuples and return early
    if (
        not isinstance(a, tuple)
        and not isinstance(b, tuple)
        and not isinstance(c, tuple)
    ):
        return a <= b and b <= c
    # next check if all inputs are same length tuples
    if isinstance(a, tuple) and isinstance(b, tuple) and isinstance(c, tuple):
        if len(a) == len(b) and len(b) == len(c):
            return all([x <= y and y <= z for x, y, z in zip(a, b, c)])
    # if neither early-exit condition is met, begin the slow path
    # put all inputs into a single tuple to make interating over them easier
    input_list: list = [a, b, c]
    # now check if there is more than one tuple, and ensure they are the same length
    last_len: int = -1
    for elem in input_list:
        if type(elem) == tuple:
            if last_len == -1:
                last_len = len(elem)
            else:
                assert len(elem) == last_len
    # now for all non-tuples, extend to target length tuple
    for i in range(3):
        if type(input_list[i]) != tuple:
            input_list[i] = tuple([input_list[i]] * last_len)

    # now compare
    return all([x <= y and y <= z for x, y, z in zip(*input_list)])


# endregion
