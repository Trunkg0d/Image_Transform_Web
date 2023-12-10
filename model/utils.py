import numpy as np

def linear_transform(img, weight=1, bias=0):
    bias = float(bias)
    weight = float(weight)
    res = img.copy().astype(np.float64)
    res = res * weight + bias
    res = res.astype(np.uint64)
    # scale
    res = np.clip(res, 0, 255)

    return res

def nonlinear_transform(img, key="log", const=1):
    res = img.copy().astype(np.float64)
    if key == "log":
        res = const * np.log2(res + 1)
    elif key == "exp":
        mean = np.mean(res)
        std_dev = np.std(res)
        coef = np.log(255)
        res = np.exp((res - mean) * coef/ std_dev)

    res = res.astype(np.uint64)
    # scale
    res = np.clip(res, 0, 255)

    return res

def calculate_histogram(img):
    # init histogram
    histogram = np.zeros(256, dtype=np.float64)

    # calculate histogram
    for pixel in img.flatten():
        histogram[pixel] += 1

    return histogram

def hist_equalization_transform(img):
    # calculate histogram
    hist = calculate_histogram(img)
    # calculate cumulative distribution function
    cdf = hist.cumsum()
    # normalize cdf
    cdf_normalized = np.floor(cdf * 255 / (cdf.max())) # M * N = cdf.max() (number of pixel)

    # result
    res = cdf_normalized[img]
    res = res.astype(np.uint64)

    return res

def find_nearest(pixel, histogram):
    """Find the nearest value of the pixel in histogram"""
    diff = histogram - pixel
    mask = np.ma.less_equal(diff, -1) # mask the negative differences
    if np.all(mask):
        index = np.abs(diff).argmin()
        return index # return min index of the nearest if target is greater than any value
    masked_diff = np.ma.masked_array(diff, mask)
    return masked_diff.argmin()

def hist_specification_transform(img, specified_img):
    original_img = img
    # calculate histograms
    original_hist = calculate_histogram(original_img)
    specified_hist = calculate_histogram(specified_img)
    # cumulative histograms
    original_cdf = original_hist.cumsum()
    specified_cdf = specified_hist.cumsum()
    # normalize cdf
    original_cdf_normalized = np.floor(original_cdf * 255 / original_cdf.max())
    specified_cdf_normalized = np.floor(specified_cdf * 255 / specified_cdf.max())

    # create mapping
    mapping = []
    for value in original_cdf_normalized:
        idx = find_nearest(value, specified_cdf_normalized)
        mapping.append(idx)
    mapping = np.array(mapping)

    res = mapping[original_img]
    res = res.astype(np.uint64)
    return res