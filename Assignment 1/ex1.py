import mediapy as media
import numpy as np

bins_number = 256

def main(video_path, video_type):
    """
    Main entry point for ex1
    :param video_path: path to video file
    :param video_type: category of the video (either 1 or 2)
    :return: a tuple of integers representing the frame number for which the scene cut was detected (i.e. the last frame index of the first scene and the first frame index of the second scene)
    """

    # imports the video and convert it to grayscale
    video = media.read_video(video_path, output_format="gray")

    # Creates an empty list of histograms
    histograms = []

    # Loops over all frames, calculate the cumulative histogram for each
    # and add it to the list
    for frame in video:
        hist = np.histogram(frame, bins=bins_number)[0]
        hist = np.cumsum(hist)

        histograms.append(hist)

    # Creates an empty list of deltas
    deltas = []

    # For every 2 consecutive frames, calculate the delta (l1) and add to the list
    for i in range(1, len(histograms)):
        delta = np.sum(np.abs(histograms[i] - histograms[i-1]))
        deltas.append(delta)

    # Find the index of the maximum delta, which is the scene cut
    # Assuming there's only one scene cut
    max_delta_index = np.argmax(deltas)

    # Return the tuple of the last frame of the first scene and the first frame of the second scene
    return (max_delta_index, max_delta_index+1)