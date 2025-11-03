import numpy as np
import matplotlib.pyplot as plt


def pltDiskProjection(center, radius, hollow = False, innerRadius = None, color = 'black', linestyle = '-', axis = None):
    theta = np.linspace(0, 2*np.pi, 500)
    xc = radius * np.cos(theta) + center[0]
    yc = radius * np.sin(theta) + center[1]
    if hollow:
        if innerRadius is None:
            raise ValueError("innerRadius must be set for a hollow disk")
        xc_in = innerRadius * np.cos(theta) + center[0]
        yc_in = innerRadius * np.sin(theta) + center[1]
    if axis is None:
        plt.plot(xc, yc, color = color, linestyle = linestyle)
        if hollow:
            plt.plot(xc_in, yc_in, color = color, linestyle = linestyle)
    else:
        axis.plot(xc, yc, color = color, linestyle = linestyle)
        if hollow:
            axis.plot(xc_in, yc_in, color = color, linestyle = linestyle)
        

def pltRectangleProjection(center, width, height, hollow = False, innerWidth = None, color = 'black', linestyle = '-', axis = None):
    left = center[0] - 0.5 * width
    right = center[0] + 0.5 * width
    hmin = center[1] - 0.5 * height
    hmax = center[1] + 0.5 * height
    if hollow:
        if innerWidth is None:
            raise ValueError("innerWidth must be set for a hollow disk")
        ileft = center[0] - 0.5 * innerWidth
        iright = center[0] + 0.5 * innerWidth
    if axis is None:
        if not hollow:
            plt.plot([left, right], [hmax, hmax], color = color, linestyle = linestyle)
            plt.plot([left, right], [hmin, hmin], color = color, linestyle = linestyle)
        plt.plot([left, left], [hmin, hmax], color = color, linestyle = linestyle)
        plt.plot([right, right], [hmin, hmax], color = color, linestyle = linestyle)
        if hollow:
            plt.plot([left, ileft], [hmax, hmax], color = color, linestyle = linestyle)
            plt.plot([left, ileft], [hmin, hmin], color = color, linestyle = linestyle)
            plt.plot([iright, right], [hmax, hmax], color = color, linestyle = linestyle)
            plt.plot([iright, right], [hmin, hmin], color = color, linestyle = linestyle)
            plt.plot([ileft, ileft], [hmin, hmax], color = color, linestyle = linestyle)
            plt.plot([iright, iright], [hmin, hmax], color = color, linestyle = linestyle)
    else:
        axis.plot([left, right], [hmax, hmax], color = color, linestyle = linestyle)
        axis.plot([left, right], [hmin, hmin], color = color, linestyle = linestyle)
        axis.plot([left, left], [hmin, hmax], color = color, linestyle = linestyle)
        axis.plot([right, right], [hmin, hmax], color = color, linestyle = linestyle)
        if hollow:
            axis.plot([left, ileft], [hmax, hmax], color = color, linestyle = linestyle)
            axis.plot([left, ileft], [hmin, hmin], color = color, linestyle = linestyle)
            axis.plot([iright, right], [hmax, hmax], color = color, linestyle = linestyle)
            axis.plot([iright, right], [hmin, hmin], color = color, linestyle = linestyle)
            axis.plot([ileft, ileft], [hmin, hmax], color = color, linestyle = linestyle)
            axis.plot([iright, iright], [hmin, hmax], color = color, linestyle = linestyle)

 
        
