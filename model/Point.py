class Point:
    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y

    def __repr__(self) -> str:
        return "(" + str(self.x) + " " + str(self.y) + ")"

    def set_val(self, x, y):
        self.x = x
        self.y = y
