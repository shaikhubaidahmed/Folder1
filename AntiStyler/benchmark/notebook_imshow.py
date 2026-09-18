"""Optional visualization helper, split out of antistyler_core.py so that
importing the core AntiStyler classes for batch evaluation never requires
matplotlib or a display backend. Only used when AntiStyler.apply(...,
visualize=True) is passed explicitly (never during the Table 1 batch run).
"""


def imshow(tensor, title=None):
    import matplotlib.pyplot as plt

    image = tensor.detach().cpu().squeeze(0).permute(1, 2, 0).clamp(0, 1)
    plt.figure()
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.axis("off")
    plt.show()
