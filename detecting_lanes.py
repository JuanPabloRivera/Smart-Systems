import cv2
import numpy as np

def find_intersection(p1, p2, p3, p4):
    a1 = p2[1] - p1[1]
    b1 = p1[0] - p2[0]
    c1 = a1*p1[0] + b1*p1[1]
    
    a2 = p4[1] - p3[1]
    b2 = p3[0] - p4[0]
    c2 = a2*p3[0] + b2*p3[1]
    
    determinant = a1*b2 - a2*b1
    # Determinant not zero
    if determinant:
        x = round((b2*c1 - b1*c2)/determinant)
        y = round((a1*c2 - a2*c1)/determinant)
        return (x,y)
    else:
        return None
    
def get_angle(p1, p2, p3, p4, degrees=False):
    vec1 = (p2[0]-p1[0], p2[1]-p1[1])
    vec2 = (p4[0]-p3[0], p4[1]-p3[1])
    
    uvec1 = vec1 / np.linalg.norm(vec2)
    uvec2 = vec2 / np.linalg.norm(vec2)
    angle = np.arccos(np.clip(np.dot(uvec1, uvec2), -1.0, 1.0))
    
    if degrees:
        angle = angle*180/np.pi
        
    return angle

# Defining the color space for the threshold
color_space = 'HSV'

# Reading threshhold values for the mask
threshold_vals = []
threshold_file = open(f'{color_space}_values', 'r')
for line in threshold_file.readlines():
    threshold_vals.append(np.array([int(x) for x in line.strip().split()]))
print(threshold_vals)

