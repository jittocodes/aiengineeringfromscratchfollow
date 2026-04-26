from vector import Vector

class Matrix:

    def __init__(self, rows):
        self.rows = [list(rows) for row in rows]
        self.shape = (len(self.rows), len(self.rows[0]))

    def __matmul__(self, other):
        if isinstance(other, Vector):
            return Vector([
                sum(self.rows[i][j] * other.components[j] for j in range(self.shape[1])) for i in range(self.shape[0])
            ])