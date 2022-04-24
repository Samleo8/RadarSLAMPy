import numpy as np
from trajectoryPlotting import Trajectory

class Map():

    def __init__(self, sequenceName: str, estTraj: Trajectory,
                 imgPathArr: list[str], filePaths: dict[str]) -> None:
        self.sequenceName = sequenceName

        self.imgPathArr = imgPathArr
        self.sequenceSize = len(self.imgPathArr)

        self.filePaths = filePaths

        self.estTraj = estTraj

    def isGoodKeyframe(self, frame, R, h):
        pass

    def addKeyframe(self, frame, R, h):
        pass