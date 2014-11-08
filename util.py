# Utilities for use by the Combiner and Crossvalidate classes.

from __future__ import division
import sys
import numpy
import math
import random

__author__ = "Brian Tomasik"
__date__ = "April/May 2009"

def info(str, newline=True):
    """
    Print str to std err.
    """
    if newline:
        sys.stderr.write("%-30s\n" % str)
    else:
        sys.stderr.write("%-30s\r" % str)

def ordered_list_intersect(list1, list2):
    """
    Return the intersection of list1 and list2, where the items returned have
    the same order as in list1.
    """
    return [x for x in list1 if x in list2]
    
def is_subset(A, B):
    """
    Return True iff A is a subset of B.
    """
    return B.union(A)==B

def remove_trailing_string(main_str, trailing_str):
    """
    Return main_str with trailing_str removed from the right.
    """
    (stripped_str, empty_str) = main_str.split(trailing_str)
    assert empty_str=="", "Bad parsing of main string."
    return stripped_str

def stderr(list):
    """
    Return the std err of a list of numbers.
    """
    return numpy.std(list) / numpy.sqrt(len(list))

def summary_stats(list):
    """
    Return a quadruple: (mean, stderr, median, length of list). Return
    ("NA", "NA", "NA", len(list)) if values not numeric.
    """
    try:
        return (numpy.mean(list), stderr(list), numpy.median(list), len(list))
    except TypeError:
        return ("NA", "NA", "NA", len(list))

def partition(list, n):
    """
    Given a list and a desired size n, return a list of partitions of
    the original list of size at most n.
    """
    n_partitions = int(math.ceil(len(list) / n))
    list_of_lists = []
    for i in range(n_partitions):
        list_of_lists.append(list[i*n:(i+1)*n])
    return list_of_lists

def logit(probability, small_num=0.0000001):
    assert probability >= 0, "probability is negative: %s" % str(probability)
    assert probability <= 1, "probability is > 1: %s" % str(probability)
    if probability < small_num:
        probability = small_num
    if probability > 1-small_num:
        probability = 1-small_num
    return math.log(probability / (1-probability))

def random_subset(orig_set, n_keep):
    if len(orig_set) <= n_keep:
        return orig_set
    """
    Old, long way of doing it....
    indices = range(len(orig_set))
    random.shuffle(indices)
    indices_keep = indices[:n_keep]
    orig_set_as_list = list(orig_set)
    return set([orig_set_as_list[i] for i in indices_keep])
    """
    return set(random.sample(orig_set, n_keep))

def write_file(filename, contents):
    file = open(filename, "w")
    file.write(contents)
    file.close()

def num_nonzeros(list):
    """
    How many nonzero elements are in this list?
    """
    return len([1 for val in list if val != 0])


def mean_if_numeric(list):
    """
    Return the mean of a numeric list. If list isn't numeric, return "NA".
    """
    try:
        return numpy.mean(list)
    except TypeError:
        return "NA"
