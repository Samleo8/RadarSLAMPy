import numpy as np


class CartCoord(object):
    '''Creates a point on a Cartesian coordinate plane with values x and y.'''

    def __init__(self, x: float, y: float):
        '''Defines x and y variables'''
        self.x = x
        self.y = y

    def __str__(self):
        return "CartCoord(%s,%s)" % (self.getX(), self.getY())

    def addCoord(self, other) -> None:
        self.increment(other.getX(), other.getY())

    def add(self, dx: float, dy: float) -> None:
        self.x += dx
        self.y += dy

    def sub(self, dx: float, dy: float) -> None:
        self.x -= dx
        self.y -= dy

    def scale(self, scaleFactor: float) -> None:
        self.scaleX(scaleFactor)
        self.scaleY(scaleFactor)

    def scaleX(self, scaleFactor: float) -> None:
        self.x *= scaleFactor

    def scaleY(self, scaleFactor: float) -> None:
        self.y *= scaleFactor

    def getX(self) -> float:
        return self.x

    def getY(self) -> float:
        return self.y

    def getDistance(self, other) -> float:
        dx = self.x - other.x
        dy = self.y - other.y

        return np.sqrt(dx**2 + dy**2)

    def getAngle(self, other):
        dx = self.x - other.x
        dy = self.y - other.y

        return np.arctan2(dy, dx)

    def asTuple(self) -> tuple:
        return (self.getX(), self.getY())

class PolarCoord(object):
    '''Creates a point on a Cartesian coordinate plane with values x and y.'''

    def __init__(self, r: float, theta: float):
        '''Defines x and y variables'''
        self.r = r
        self.theta = theta

    def __str__(self):
        return "PolarCoord(%s,%s)" % (self.getR(), self.getTheta())

    def getR(self) -> float:
        return self.r

    def getTheta(self) -> float:
        return self.theta

    def scaleR(self, scaleFactor) -> None:
        self.r *= scaleFactor

    def toCart(self) -> CartCoord:
        x = self.r * np.cos(self.theta)
        y = self.r * np.sin(self.theta)

        return CartCoord(x, y)

    def asTuple(self) -> tuple:
        return (self.getR(), self.getTheta())