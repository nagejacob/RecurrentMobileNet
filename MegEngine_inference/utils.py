def aug(img, flip_h, flip_w, transpose):
    if flip_h:
        img = img[::-1, :]
    if flip_w:
        img = img[:, ::-1]
    if transpose:
        img = img.T
    return img

def unaug(img, flip_h, flip_w, transpose):
    if transpose:
        img = img.T
    if flip_w:
        img = img[:, ::-1]
    if flip_h:
        img = img[::-1, :]
    return img