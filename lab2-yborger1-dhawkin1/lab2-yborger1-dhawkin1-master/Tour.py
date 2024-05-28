########################################
# CS63: Artificial Intelligence, Lab 2
# Fall 2022, Swarthmore College
########################################
# NOTE: you should not need to modify this file.
########################################

class Tour(tuple):
    """Represents a traveling salesperson tour by a tuple of city names.
    The tuple lists cities in the order visited by the tour; the last leg
    of the tour goes from the final city back to the first.

    __init__ is inherited from tuple, and takes an iterable
    (list, tuple, etc.)
    """
    def __repr__(self):
        return "-->".join(self) + "-->" + self[0]

    def move_city(self, i, j):
        """Gives a new Tour with the city at index i moved to index j."""
        if i < j:
            t = self[:i] + self[i+1:j+1] + self[i:i+1] + self[j+1:]
        else:
            t = self[:j] + self[i:i+1] + self[j:i] + self[i+1:]
        return Tour(t)

    def add_city(self, city, index):
        """Gives a new Tour with city inserted at index."""
        return Tour(self[:index] + (city,) + self[index:])

    def remove_city(self, city):
        """Gives a new Tour with city removed."""
        index = self.index(city)
        return Tour(self[:index] + self[index+1:])

if __name__ == '__main__':
    # Create a tour of local cities
    tour1 = Tour(["Swarthmore", "Media", "Wallingford", "Broomall"])
    print(tour1)
    # Create a new tour by moving city at index 0 to index 2
    tour2 = tour1.move_city(0,2)
    print(tour2)
