class Vector:
    def __init__(self, components):
        self.components = components
        self.dim = len(self.components)

    def __add__(self, other):
        return Vector([a + b for a,b in zip(self.components, other.components)])
    
    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.components, other.components)])
    
    def dot(self, other):
        return sum(a * b for a, b in zip(self.components, other.components))
    
    def magnitude(self):
        return sum(x ** 2 for x in self.components) ** 0.5
    
    def normalize(self):
        mag = self.magnitude()
        return Vector([x / mag for x in self.magnitude])
    
    def cosine_similarity(self, other):
        return self.dot(other) / (self.magnitude() * other.magnitude())
    
    def __repr__(self):
        return f"Vector({self.components})"
    
a = Vector([1, 2, 3])
b = Vector([4, 5, 6])


print(f"a + b {a + b}")
print(f"a - b: f{a - b}")
print(f"a . b {a.dot(b)}")
print(f"|a| = {a.magnitude():.4f}")
print(f"Cosine Sim: {a.cosine_similarity(b):.4f}")