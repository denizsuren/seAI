def count_spaces(sentence: str) -> int:
    """Return the number of space characters in the given sentence.

    Args:
        sentence: The input string.
    Returns:
        The count of spaces (character ' ') in the string.
    """
    # Simple count using str.count
    return sentence.count(' ')

# Simple manual test when run as script
if __name__ == "__main__":
    test_sentences = [
        "Hello world",
        "NoSpacesHere",
        "  Leading and trailing  ",
        "Multiple   spaces",
        "",
    ]
    for s in test_sentences:
        print(f"'{s}' -> {count_spaces(s)} spaces")
