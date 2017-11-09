class Circle:
    def __init__(self, circle=[]):
        self.circle = circle

    def add_point(self, p):
        self.circle.append(p)

    def get_circle(self):
        return self.circle

    def clear_circle(self):
        self.circle = []

    def __repr__(self) -> str:
        s = "("
        if len(self.circle)>0:
            for point in self.circle:
                s += point.__repr__()
                s += ","
            s = s[:-1]
        s += ")"
        return s