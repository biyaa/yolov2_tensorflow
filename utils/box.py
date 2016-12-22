def overlap(x1, w1, x2, w2):
    l1 = x1 - w1/2
    l2 = x2 - w2/2
    left = l1 if l1 > l2 else l2
    r1 = x1 + w1/2
    r2 = x2 + w2/2
    right = r1 if r1 < r2 else r2
    return right - left

def box_intersection(a, b):
    w = overlap(a[0], a[2], b[0], b[2])
    h = overlap(a[1], a[3], b[1], b[3])
    if w < 0 or h < 0: 
        return 0
    area = w*h
    return area

def  box_union(a, b):
    i = box_intersection(a, b)
    u = a[2]*a[3] + b[2]*b[3] - i
    return u

def  box_iou(a, b):
    return box_intersection(a, b)/box_union(a, b)

