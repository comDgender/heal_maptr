import numpy as np
import torch
from PIL import Image

from heal.visualize.det import DetViz


def test_det_viz_saves_image_with_pred_bboxes(tmp_path):
    viz = DetViz(is_plot=True, save_path=str(tmp_path), score_thresh=0.1)
    preds = {
        "pred_bboxes": [
            torch.tensor(
                [
                    [10.0, 10.0, 50.0, 50.0, 0.9, 1.0],
                    [60.0, 60.0, 90.0, 90.0, 0.2, 2.0],
                ]
            )
        ]
    }
    image = np.full((100, 100, 3), 180, dtype=np.uint8)
    inputs = {"img": Image.fromarray(image, mode="RGB")}

    viz(preds, inputs)

    output = tmp_path / "det_pred_0.png"
    assert output.is_file()
    assert output.stat().st_size > 1024
    saved = np.array(Image.open(output))
    assert saved.std() > 1.0
    assert viz.sample_idx == 1
