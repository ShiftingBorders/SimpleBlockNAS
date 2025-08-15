class GAException(Exception):
    pass

class TripletLengthException(GAException):
    """
    Exception raised when a triplet has fewer than 3 elements.

    This exception is thrown during crossover operations when the input triplets
    do not have the expected length of 3 elements, which is required for
    proper crossover operations.
    """
    pass