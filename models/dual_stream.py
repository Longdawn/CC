"""DualStreamDetector placeholder combining two streams."""
class DualStreamDetector:
    def __init__(self, cfg=None):
        self.cfg = cfg

    def forward(self, x1, x2):
        raise NotImplementedError("DualStream forward not implemented")
