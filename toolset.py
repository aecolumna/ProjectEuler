#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

# This file is used as both a shell script and as a Python script.

# acolumna

import operator
import itertools
import functools
import collections
import random

from math import sqrt, log, factorial
from itertools import combinations, islice, permutations
import numpy as np

from numba import jit, jit

from operator import mul, add, itemgetter, methodcaller
import operator as op

try:
    from functools import reduce, wraps, lru_cache
except ImportError:
    pass


# rename cartesian product
cartesian_product = itertools.product


@lru_cache(maxsize=None)
def is_prime(n):
    """Return True if n is a prime number (1 is not considered prime)."""
    if n < 3:
         return n == 2
    if not n % 2:
        return False
    if any(n % x == 0 for x in range(3, int(sqrt(n)) + 1, 2)):
        return False
    return True

def gen_primes(lim):
    """Yield primes until lim. Lower on memory """
    yield 2
    yield 3
    for i in range(5, lim + 1, 2):
        if is_prime(i):
            yield i

def primes(n = 1024):
    """generate n primes (Sieve of Erastothenes)."""
    yield 2
    yield 3

    multiples = set(chain(range( 2, n + 1, 2), range( 3, n + 1, 6)))

    for i in range(5, n + 1, 2):  # check on
        if i not in multiples:
            yield i
            multiples.update(range(i * i, n + 1, i))

#prime factorization


def Mathematica(string):
    import subprocess
    return subprocess.check_output(
        [ "run", string], universal_newlines=True)

def prime_factors(N):
    """Factor N into its components.
    THM: composite number must have a prime factor <= sqrt(n).
    Otherwise N is prime
    """
    if N <= 1:
        return None

    max_factor = int(N ** 0.5)

    flag = False

    for i in gen_primes(max_factor):

        while N % i == 0:
            yield i
            N //= i
            flag = True

        if flag:
            flag = False
            max_factor = int( N**0.5)

        if i > max_factor:
            break

    if N != 1:
        yield N


def fibonacci(N, n_1 = 1, n = 0):
    """generate n fibonacci numbers"""
    for i in range(N):
        yield n
        n_1, n = n_1 + n, n_1


def gen_fibs(n1 = 1, n = 0):
    """generate fibonacci sequence by Andres. Infinite amount by default"""
    while True:
        yield n1
        n1, n = n1 + n, n1


def product(arr, reduce = reduce, mul = operator.mul):
    """Product of elements in list"""
    return reduce(mul, arr, 1)

try:
    Π = product
    π = np.pi
    Σ = sum
    plancks = 6.62606957e-34
    ℏ = plancks / (np.pi * 2)
except:
    pass


def dotproduct(vec1, vec2, sum = sum, mul = operator.mul, map = map):
    """example of how to optimize functions by making lookups local
    return: int
    Note, many of the above recipes can be optimized by replacing
    global lookups with local variables defined as default values.
    For example, the dotproduct recipe can be written as:
    """
    return sum(map(mul, vec1, vec2))


def consume(iterator, n = None):
    """Advance the iterator n-steps ahead. If n is none, consume entirely. By Andres """
    collections.deque(islice(iterator, n), maxlen = 0)

#sliding window, instead of deck!
def n_grams(a, n):
    z = (islice(a, i, None) for i in range(n))
    return zip(*z)

def swap(arr, i1, i2):
    """ Swaps values between arrays """
    arr[i1], arr[i2] = arr[i2], arr[i1]


def index(n, iterable):
    "Returns the nth item"
    return next(islice(iterable, n, n + 1))

from itertools import chain
from types import GeneratorType

def powerset(lst):
    """powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"""
    if isinstance(lst, GeneratorType):
        lst = list(lst)
    return chain.from_iterable(combinations(lst, r) for r in range(1, len(lst) + 1))

def pairwise(arr):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(arr)
    next(b, None)
    return zip(a, b)

def flatten(listOfLists):
    "Flatten one level of nesting"
    return chain.from_iterable(listOfLists)

def take_every(iterable, n):
    """Take an element from iterator every n elements"""
    return islice(iterable, 0, None, n)


def compose(f, g):
    """Compose two functions -> compose(f, g)(x) -> f(g(x))"""

    def _wrapper(*args, **kwargs):
        return f(g(*args, **kwargs))

    return _wrapper

def divisors(n):
    """Return all divisors of n: divisors(12) -> 1,2,3,6,12"""
    all_factors = [[f ** p for p in range(fp + 1)] for (f, fp) in factorize(n)]
    return (product(ns) for ns in cartesian_product(*all_factors))


def factorize(num):
    """Factorize a number returning occurrences of its prime factors"""
    return ((factor, len(fs)) for (factor, fs) in itertools.groupby(prime_factors(num)))


def greatest_common_divisor(a, b):
    """Return greatest common divisor of a and b"""
    return greatest_common_divisor(b, a % b) if b else a


def least_common_multiple(a, b):
    """Return least common multiples of a and b"""
    return (a * b) / greatest_common_divisor(a, b)

def rrange(size = 10, interval = 100):
    """returns random numbers into array"""
    for i in range(size):
        yield random.randint(0, interval)


def first(iterable):
    """Take first element in the iterable"""
    return iterable.next()

def last(iterable):
    """Take last element in the iterable"""
    return reduce(lambda x, y: y, iterable)

#######Decorators

class tail_recursive:
    """Tail recursive decorator."""
    # By Michele Simionato
    CONTINUE = object()  # sentinel

    def __init__(self, func = op.add):
        self.func = func
        self.firstcall = True
        functools.update_wrapper(self, func)

    def __call__(self, *args, **kwd):
        try:
            if self.firstcall:  # start looping
                self.firstcall = False
                while True:
                    result = self.func(*args, **kwd)
                    if result is self.CONTINUE:  # update arguments
                        args, kwd = self.argskwd
                    else:  # last call
                        break
            else:  # return the arguments of the tail call
                self.argskwd = args, kwd
                return self.CONTINUE
        except:  # reset and re-raise
            self.firstcall = True
            raise
        else:  # reset and exit
            self.firstcall = True
            return result
def vector(func):
    @wraps(func)
    def new_func(iterable):
        try:
            for i, x in enumerate(iterable):
                iterable[i] = func(x)
        except:
            return (func(i) for i in iterable)

    return new_func
'''
itertools.compress(data, selectors)
filter NONE values from data
# compress('ABCDEF', [1,0,1,0,1,1]) --> A C E F

def chain(*iterables):
    # chain('ABC', 'DEF') --> A B C D E F

permutations(iterable, r=None):
    # permutations('ABCD', 2) --> AB AC AD BA BC BD CA CB CD DA DB DC
    # permutations(range(3)) --> 012 021 102 120 201 210

combinations_with_replacement(iterable, r):
    # combinations_with_replacement('ABC', 2) --> AA AB AC BB BC CC

takewhile(predicate, iterable)
    Make an iterator that returns elements from the iterable as long as the predicate is true.
    # takewhile(lambda x: x<5, [1,4,6,4,1]) --> 1 4

dropwhile(predicate, iterable):
    # dropwhile(lambda x: x<5, [1,4,6,4,1]) --> 6 4 1

itertools.tee(iterable, n=2)
    Return n independent iterators from a single iterable.

itertools.starmap(function, iterable):
    # itertools.starmap(pow, [(2,5), (3,2), (10,3)]) --> 32 9 1000

class itertools.grouby:
    # [k for k, g in itertools.grouby('AAAABBBCCDAABBB')] --> A B C D A B
    # [list(g) for k, g in itertools.grouby('AAAABBBCCD')] --> AAAA BBB CC D

def compact(it):
    """Filter None values from iterator"""
    return filter(bool, it)
'''
