## Inference from Himalaya's combined (strokes + image) model using OpenCV
- Link to Kaggle: https://www.kaggle.com/code/yuvrajraghuvanshis/himalaya-s-combined-model

## Building up
- Earlier we were feeding too many strokes even though the training dataset is capped to 130 strokes points
- Truncating points was as stupid as feeding as is because of information loss
- A better sampling ([Douglas-Peucker](https://en.wikipedia.org/wiki/Ramer-Douglas-Peucker_algorithm)) was used to sample points
- Also, pen state being 0 implies that pen is down (I still can't reason why, anyway, we were using it backwards)


## inference_v1.py
- Based on initial model same model
- Here the were processed like: 
    ```python
    def preprocess_stroke(stroke, max_len=MAX_SEQ_LEN):
    stroke = stroke.astype(np.float32)
    stroke[:, 0] = np.cumsum(stroke[:, 0])  # make x like absolute from delta
    stroke[:, 1] = np.cumsum(stroke[:, 1])  # same do for y also
    stroke[:, 0] -= stroke[:, 0].mean()  # center x to 0
    stroke[:, 1] -= stroke[:, 1].mean()  # center y to 0 again
    if len(stroke) > max_len:
        return stroke[:max_len]  # if too long, cut itt
    pad = np.zeros((max_len - len(stroke), STROKE_FEATURES), dtype=np.float32)
    return np.vstack([stroke, pad])  # else just fill with zero
    ```
- The issue was the strokes are centered but not scaled.
- I didn't pay attention to it earlier and tried all sorts of hacks like manually checking the range of first 50 samples and scaling model input accordingly

## inference.py
- Here the first change I did was to update the `preprocess_stroke` to also scale.
- This made it consistent and predictable so updated the inference.py as well to feed same data


## Issues
- Not an issue per say but currently the model stroke input is flipped, although it does not seem to affect the accuracy
- Flower image was not being predicted. Issue: From my earlier analysis I found that on an average only 50 points were being used despite the max limit of 130. So, I capped the points limit to 50 fearing that increasing might lead to accuracy loss. Fix: Now as Dougles-Peuckers algo is working fine the max limit is updated to 130