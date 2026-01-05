# Median Algorithm Benchmark

A comprehensive benchmark suite comparing different median calculation algorithms in Python.

## Algorithms Tested

1. **Pure Python (sorting)** - Built-in `sorted()` function
2. **NumPy** - `np.median()`
3. **Pandas** - `pd.Series.median()`
4. **Quickselect** - Average O(n) selection algorithm
5. **Two Heaps** - Maintains balanced min/max heaps
6. **heapq.nlargest** - File-based heap approach
7. **Median of Medians** - Guaranteed O(n) but slower in practice

## Prerequisites

### Without Docker
- Python 3.11+
- pip

### With Docker
- Docker installed and running

## Usage

### Option 1: Using Docker (Recommended)

Build the Docker image:
```bash
docker build -t median-benchmark .
```

Run the benchmark:
```bash
docker run median-benchmark
```

The container will:
- Generate a test file with 10 million random numbers
- Run all benchmark tests
- Display results and timing summary

### Option 2: Local Python Environment

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the benchmark:
```bash
python median_benchmark.py
```

## Configuration

You can modify the test size by editing `median_benchmark.py`:

```python
# Change the number of test values (default: 10,000,000)
generate_test_file(filename, n=10000000)
```

## Output

The benchmark will display:
- Test name and result for each algorithm
- Execution time for each algorithm
- Summary table sorted by performance

Example output:
```
Testing Pure Python (sorting)...
  Result: 5000123.45
  Time: 2.1234 seconds

...

SUMMARY
NumPy               : 0.5234 seconds
Quickselect         : 1.2345 seconds
Pure Python         : 2.1234 seconds
...
```

## Implementation Details

### Benchmark Decorator

The timing functionality is implemented using a decorator pattern:

```python
@benchmark('Test Name')
def my_function(data):
    # function implementation
    pass
```

The decorator automatically:
- Prints the test name
- Times execution
- Displays the result and elapsed time
- Stores timing in `function.last_elapsed`

## File Structure

```
.
├── Dockerfile              # Docker configuration
├── .dockerignore          # Files excluded from Docker build
├── requirements.txt       # Python dependencies
├── median_benchmark.py    # Main benchmark script
└── README.md             # This file
```

## Performance Notes

- **NumPy** typically offers the best performance for large datasets
- **Quickselect** provides good average-case performance
- **Median of Medians** guarantees O(n) but has high constants
- **Two Heaps** is optimized for streaming data scenarios
- Test file is generated fresh each run and stored as `numbers.txt`

## License

MIT
