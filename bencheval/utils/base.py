def is_int(x):
    """
    Check if a string can be converted to an integer.

    Args:
        x(str): Input string.

    Returns:
        bool: True if x can be converted to an integer, False otherwise
    """
    try:
        int(x)
        return True
    except ValueError:
        return False


def is_number(s):
    """
    Check if a string can be converted to a number.

    Args:
        s(str): Input string.

    Returns:
        bool: True if s can be converted to a number, False otherwise
    """
    try:
        float(s)
        return True
    except ValueError:
        return False


def get_combinations(s, k):
    """
    Generate all subsets of size k from set s.

    Args:
        s(list): List of elements to get combinations from.
        k(int): Size of each combination.

    Returns:
        list: A list of combinations, where each combination is represented as a list.
    """
    if k == 0:
        return [[]]
    elif k > len(s):
        return []
    else:
        all_combinations = []
        for i in range(len(s)):
            # For each element in the set, generate the combinations that include this element
            # and then recurse to generate combinations from the remaining elements
            element = s[i]
            remaining_elements = s[i + 1:]
            for c in get_combinations(remaining_elements, k - 1):
                all_combinations.append([element] + c)
        return all_combinations
