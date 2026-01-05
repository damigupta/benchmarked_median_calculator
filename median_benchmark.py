import time
import random
import heapq
from functools import wraps

def generate_test_file(filename, n=100000):
#def generate_test_file(filename, n=10000000):
    """Generate a test file with random numbers"""
    print(f"Generating test file with {n} numbers...")
    with open(filename, 'w') as f:
        for _ in range(n):
            f.write(f"{random.randint(1, 10000000)}\n")
    print("Test file generated.\n")


def read_numbers(filename):
    """Read numbers from file"""
    with open(filename, 'r') as f:
        return [int(line.strip()) for line in f]


def benchmark(name=None):
    """Decorator to benchmark function execution time"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            display_name = name or func.__name__
            print(f"Testing {display_name}...")
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            print(f"  Result: {result:.2f}")
            print(f"  Time: {elapsed:.4f} seconds\n")
            wrapper.last_elapsed = elapsed
            return result
        return wrapper
    return decorator


@benchmark('Pure Python (sorting)')
def median_pure_python(numbers):
    """Calculate median using pure Python (sorting)"""
    sorted_nums = sorted(numbers)
    n = len(sorted_nums)
    if n % 2 == 0:
        return (sorted_nums[n//2 - 1] + sorted_nums[n//2]) / 2
    else:
        return sorted_nums[n//2]


@benchmark('NumPy')
def median_numpy(numbers):
    """Calculate median using NumPy"""
    import numpy as np
    return np.median(numbers)


@benchmark('Pandas')
def median_pandas(numbers):
    """Calculate median using Pandas"""
    import pandas as pd
    return pd.Series(numbers).median()


def partition(arr, left, right, pivot_index):
    """Partition array for quickselect"""
    pivot_value = arr[pivot_index]
    arr[pivot_index], arr[right] = arr[right], arr[pivot_index]
    store_index = left
    for i in range(left, right):
        if arr[i] < pivot_value:
            arr[store_index], arr[i] = arr[i], arr[store_index]
            store_index += 1
    arr[right], arr[store_index] = arr[store_index], arr[right]
    return store_index


def _quickselect(arr, left, right, k):
    """Quickselect to find  kth smallest element"""
    if left == right:
        return arr[left]

    pivot_index = random.randint(left, right)
    pivot_index = partition(arr, left, right, pivot_index)

    if k == pivot_index:
        return arr[k]
    elif k < pivot_index:
        return _quickselect(arr, left, pivot_index - 1, k)
    else:
        return _quickselect(arr, pivot_index + 1, right, k)


@benchmark('Quickselect')
def median_quickselect(numbers):
    """Calculate median using quickselect"""
    arr = numbers.copy()
    n = len(arr)
    if n % 2 == 1:
        return _quickselect(arr, 0, n - 1, n // 2)
    else:
        return (_quickselect(arr, 0, n - 1, n // 2 - 1) +
                _quickselect(arr, 0, n - 1, n // 2)) / 2


@benchmark('Two Heaps')
def median_two_heaps(numbers):
    """Calculate median using two heaps - one for lower half, one for upper half.

    Keep heaps balanced so median is always at the top(s). For [1,2,3,4,5], max_heap
    has [3,2,1] and min_heap has [4,5], so median = 3.
    """
    max_heap = []  # lower half (stored as negative values)
    min_heap = []  # upper half

    for num in numbers:
        if not max_heap or num <= -max_heap[0]:
            heapq.heappush(max_heap, -num)
        else:
            heapq.heappush(min_heap, num)

        # Balance heaps
        if len(max_heap) > len(min_heap) + 1:
            heapq.heappush(min_heap, -heapq.heappop(max_heap))
        elif len(min_heap) > len(max_heap):
            heapq.heappush(max_heap, -heapq.heappop(min_heap))

    if len(max_heap) > len(min_heap):
        return -max_heap[0]
    else:
        return (-max_heap[0] + min_heap[0]) / 2


@benchmark('heapq.nlargest (from file)')
def median_from_file_heap(filename):
    """Calculate median using heapq.nlargest from file"""
    with open(filename) as f:
        numbers = [float(line) for line in f]
    return heapq.nlargest(len(numbers)//2 + 1, numbers)[-1] if len(numbers) % 2 == 1 \
           else (heapq.nlargest(len(numbers)//2, numbers)[-1] +
                 heapq.nlargest(len(numbers)//2 + 1, numbers)[-1]) / 2



def _median_of_medians_partition(arr, left, right, pivot_index):
    """Partition array so elements < pivot are on left, >= pivot on right.
    Returns final position of pivot."""
    pivot_value = arr[pivot_index]
    arr[pivot_index], arr[right] = arr[right], arr[pivot_index]
    store_index = left
    for i in range(left, right):
        if arr[i] < pivot_value:
            arr[store_index], arr[i] = arr[i], arr[store_index]
            store_index += 1
    arr[right], arr[store_index] = arr[store_index], arr[right]
    return store_index


def _median_of_medians_select(arr, left, right, k):
    """Select kth element using median of medians"""
    if left == right:
        return arr[left]

    # Divide into groups of 5
    groups = []
    for i in range(left, right + 1, 5):
        group = arr[i:min(i + 5, right + 1)]
        group.sort()
        groups.append(group[len(group) // 2])

    if len(groups) <= 5:
        pivot = sorted(groups)[len(groups) // 2]
    else:
        pivot = _median_of_medians_select(groups, 0, len(groups) - 1, len(groups) // 2)

    # Find pivot index
    pivot_index = arr.index(pivot, left, right + 1)
    pivot_index = _median_of_medians_partition(arr, left, right, pivot_index)

    if k == pivot_index:
        return arr[k]
    elif k < pivot_index:
        return _median_of_medians_select(arr, left, pivot_index - 1, k)
    else:
        return _median_of_medians_select(arr, pivot_index + 1, right, k)


@benchmark('Median of Medians')
def median_median_of_medians(numbers):
    """Calculate median using median of medians algorithm.

    Divides into groups of 5, finds median of each group, then recursively
    finds median of medians as pivot. Guarantees O(n) time but slow in practice.
    """
    arr = numbers.copy()
    n = len(arr)
    if n % 2 == 1:
        return _median_of_medians_select(arr, 0, n - 1, n // 2)
    else:
        return (_median_of_medians_select(arr, 0, n - 1, n // 2 - 1) +
                _median_of_medians_select(arr, 0, n - 1, n // 2)) / 2


if __name__ == "__main__":
    filename = "numbers.txt"

    # Generate test file
    generate_test_file(filename, n=100000)  # Adjust size as needed

    # Read numbers
    print("Reading numbers from file...")
    numbers = read_numbers(filename)
    print(f"Read {len(numbers)} numbers.\n")

    print("=" * 50)
    print("MEDIAN CALCULATION BENCHMARK")
    print("=" * 50)
    print()

    # Benchmark each method
    times = {}

    median_pure_python(numbers)
    times['Pure Python'] = median_pure_python.last_elapsed

    median_numpy(numbers)
    times['NumPy'] = median_numpy.last_elapsed

    median_pandas(numbers)
    times['Pandas'] = median_pandas.last_elapsed

    median_quickselect(numbers)
    times['Quickselect'] = median_quickselect.last_elapsed

    median_two_heaps(numbers)
    times['Two Heaps'] = median_two_heaps.last_elapsed

    median_median_of_medians(numbers)
    times['Median of Medians'] = median_median_of_medians.last_elapsed

    median_from_file_heap(filename)
    times['heapq.nlargest'] = median_from_file_heap.last_elapsed


    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for method, time_taken in sorted(times.items(), key=lambda x: x[1]):
        print(f"{method:20s}: {time_taken:.4f} seconds")

    # Returns (key, value)
    min_key, min_val = min(times.items(), key=lambda item: item[1])

    print(f"\n\nMin value is for method {min_key}, at: {min_val:.4f} seconds")
