from typing import List, Tuple

import numpy as np
import tensorflow as tf

from supervision.detection.core import Detections
from supervision.tracker.byte_tracker import matching
from supervision.tracker.byte_tracker.basetrack import BaseTrack, TrackState

from LSTMTrack.LSTM_predictor import LSTM_predictor

from sklearn.metrics.pairwise import cosine_similarity

from scipy.optimize import linear_sum_assignment

from torchreid import utils as ut


class STrack(BaseTrack):
    # shared_kalman = KalmanFilter()
    shared_LSTM_predictor = LSTM_predictor(LSTM_model, res=[vid_info.width, vid_info.height])

    def __init__(self, tlwh, feature, score, class_id):
        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float64)
        self.feature = np.asarray(feature, dtype=np.float64)
        # this is the last predicted feature vect
        # print("This is the vector:",self.feature, self._tlwh)
        # print("This is the shape of vector",self.feature.shape, self._tlwh.shape)
        self.pred_vect = tf.concat([tf.convert_to_tensor(self.feature, dtype=tf.float64), tf.convert_to_tensor(self._tlwh, dtype=tf.float64)], axis=0)
        self.LSTM_predict = None
        # self.mean, self.covariance = None, None
        # I think we want this to be outside the track. A track obj is just the
        # feature vect in this point in time.
        # or maybe we have the sequence in here and the stored locations outside
        self.sequence = None
        self.is_activated = False

        self.score = score
        self.cls = -1
        self.cls_hist = []  # (cls id, freq)
        self.update_cls(class_id, score)
        self.tracklet_len = 1

    def update_cls(self, cls, score):
        if len(self.cls_hist) > 0:
            max_freq = 0
            found = False
            for c in self.cls_hist:
                if cls == c[0]:
                    c[1] += score
                    found = True

                if c[1] > max_freq:
                    max_freq = c[1]
                    self.cls = c[0]
            if not found:
                self.cls_hist.append([cls, score])
                self.cls = cls
        else:
            self.cls_hist.append([cls, score])
            self.cls = cls

    def predict(self):
        # sequence = self.sequence.copy()
        # if self.state != TrackState.Tracked:
        #     sequence[0] = tf.empty(self.feature.shape + self._tlwh.shape)
        self.pred_vect = self.LSTM_predict.predict(self.sequence[:,-4:])
        self._tlwh = self.pred_vect[-4:]
        self.feature = self.pred_vect[:-4] # don't change it

        # plan to add to sequence
        # from the start and go longer as the sequence gets longer until
        # tracklet_len = max_seq_len, then shift tensor with new pred on the end
        if self.tracklet_len < self.LSTM_predict.max_seq_len:
          self.sequence[self.tracklet_len] = self.pred_vect
        else:
          self.sequence = np.roll(self.sequence, shift=-1, axis=0)
          self.sequence[-1] = self.pred_vect
        # make sure the stored sequence is not longer than max_seq_len
        # self.sequence = self.sequence[-self.LSTM_predict.max_seq_len:]
        self.tracklet_len += 1
        # self.mean, self.covariance = self.kalman_filter.predict(
        #     mean_state, self.covariance
        # )

    @staticmethod
    def multi_predict(stracks):
        # if len(stracks) > 0:
        #     multi_mean = []
        #     multi_covariance = []
        #     for i, st in enumerate(stracks):
        #         multi_mean.append(st.mean.copy())
        #         multi_covariance.append(st.covariance)
        #         if st.state != TrackState.Tracked:
        #             multi_mean[i][7] = 0

        #     multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(
        #         np.asarray(multi_mean), np.asarray(multi_covariance)
        #     )
        #     for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
        #         stracks[i].mean = mean
        #         stracks[i].covariance = cov

        if len(stracks) > 0:
            multi_seq = np.zeros([len(stracks), STrack.shared_LSTM_predictor.max_seq_len, STrack.shared_LSTM_predictor.tot_vect_len])
            # multi_seq = np.zeros([len(stracks), STrack.shared_LSTM_predictor.max_seq_len, 4])
            multi_track_len = np.empty([len(stracks)], dtype=np.int32)
            for i, st in enumerate(stracks):
                # multi_sequence with 0 in front
                multi_seq[i] = st.sequence.copy()
                # multi_seq[i] = st.sequence.copy()[:,-4:]
                multi_track_len[i] = int(st.tracklet_len)
                # print("This is the sequence we want to pred in multi_pred: ", multi_seq[i], multi_track_len[i])

            multi_pred = STrack.shared_LSTM_predictor.multi_predict(multi_seq, multi_track_len)
            for i, pred_vect in enumerate(multi_pred):
                track = stracks[i]
                track.pred_vect = pred_vect
                track._tlwh = stracks[i].pred_vect[-4:]
                track.feature = stracks[i].pred_vect[:-4]
                if track.tracklet_len < stracks[i].LSTM_predict.max_seq_len:
                  track.sequence[track.tracklet_len] = track.pred_vect
                  # track.sequence[track.tracklet_len] = np.concatenate([track.feature, track.pred_vect], axis=0)
                else:
                  # i think I am rolling the wrong way, trying shift = 1
                  # print(f"preroll: {track.sequence}")
                  track.sequence = np.roll(track.sequence, shift=-1, axis=0)
                  # print(f"postroll: {track.sequence}")
                  track.sequence[-1] = pred_vect
                  # track.sequence[-1] = np.concatenate([track.feature, track.pred_vect], axis=0)
                track.tracklet_len += 1
                # plan to add to sequence
                # from the start and go longer as the sequence gets longer until
                # tracklet_len = max_seq_len, then shift tensor with new pred on
                # the end. Need to update the whole sequence here as well.
    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
      if len(stracks) > 0:
        multi_seq = np.zeros([len(stracks), STrack.shared_LSTM_predictor.max_seq_len, STrack.shared_LSTM_predictor.tot_vect_len])

        for i, st in enumerate(stracks):
          # multi_sequence with 0 in front
          # multi_seq[i] = st.sequence.copy()

          bbox = st._tlwh
          # print(f"x, y initial: {bbox[:2]}")
          # print(f"w, h initial: {bbox[2:]}")
          R = H[:2, :2]
          t = H[:2, 2]
          # print(f"R: {R}, t: {t}")
          bb_pos_adj = R.dot(bbox[:2]) + t
          bb_dim_adj = R.dot(bbox[2:]) + t
          st._tlwh = np.concatenate(bb_pos_adj, bb_dim_adj)
          if st.tracklet_len < st.LSTM_predict.max_seq_len:
            st.sequence[st.tracklet_len-1] = tf.concat([st.feature, st._tlwh], axis=0) # change the last predicted val to the true val
          else:
            st.sequence[-1] = tf.concat([st.feature, st._tlwh], axis=0) # change the last predicted val to the true val


    def activate(self, LSTM_predictor, frame_id):
        # """Start a new tracklet"""
        # self.kalman_filter = kalman_filter
        # self.track_id = self.next_id()
        # self.mean, self.covariance = self.kalman_filter.initiate(
        #     self.tlwh_to_xyah(self._tlwh)
        # )

        # self.tracklet_len = 0
        # self.state = TrackState.Tracked
        # if frame_id == 1:
        #     self.is_activated = True
        # self.frame_id = frame_id
        # self.start_frame = frame_id

        """Start a new tracklet"""
        self.LSTM_predict = LSTM_predictor
        self.track_id = self.next_id()
        # maybe extract the features here?
        # self.feature = self.LSTM_predict.initiate(self._tlwh, self.feature)

        self.sequence = np.zeros([self.LSTM_predict.max_seq_len, self.LSTM_predict.tot_vect_len])
        self.sequence[0] = self.pred_vect
        # print("sequence after activation: ", self.sequence)

        self.tracklet_len = 1
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        # self.mean, self.covariance = self.kalman_filter.update(
        #     self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        # )
        # self.tracklet_len = 0
        # self.state = TrackState.Tracked
        # self.is_activated = True
        # self.frame_id = frame_id
        # if new_id:
        #     self.track_id = self.next_id()
        # self.score = new_track.score

        # need to figure out what to do here
        # might need to extract features here
        self.feature = new_track.feature
        self._tlwh = new_track._tlwh
        if self.tracklet_len < self.LSTM_predict.max_seq_len:
          self.sequence[self.tracklet_len-1] = tf.concat([self.feature, self._tlwh], axis=0) # change the last predicted val to the true val
        else:
          self.sequence[-1] = tf.concat([self.feature, self._tlwh], axis=0) # change the last predicted val to the true val

        # self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

        self.update_cls(new_track.cls, new_track.score)


    # instead of doing kalman update, we simply want to add either the prediction
    # or the matched box to the sequence
    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        # self.tracklet_len += 1

        new_tlwh = new_track._tlwh
        self._tlwh = new_tlwh
        self.feature = new_track.feature
        if self.tracklet_len < self.LSTM_predict.max_seq_len:
          self.sequence[self.tracklet_len-1] = tf.concat([self.feature, self._tlwh], axis=0) # change the last predicted val to the true val
        else:
          self.sequence[-1] = tf.concat([self.feature, self._tlwh], axis=0) # change the last predicted val to the true val

        # self.mean, self.covariance = self.kalman_filter.update(
        #     self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh)
        # )
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        self.update_cls(new_track.cls, new_track.score)


    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.
        """
        return self._tlwh.copy()

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return "OT_{}_({}-{})".format(self.track_id, self.start_frame, self.end_frame)


def detections2boxes(detections: Detections, features: np.ndarray) -> np.ndarray:
    """
    Convert Supervision Detections to numpy tensors for further computation.
    Args:
        detections (Detections): Detections/Targets in the format of sv.Detections.
        features (ndarray): The corresponding image features of each detection.
        Has shape [N, num_features]
    Returns:
        (np.ndarray): Detections as numpy tensors as in
            `(x_min, y_min, x_max, y_max, confidence, class_id, feature_vect)` order.
    """
    return np.hstack(
        (
            detections.xyxy,
            detections.confidence[:, np.newaxis],
            detections.class_id[:, np.newaxis],
            features
        )
    )


def box_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])

def box_iou_batch(boxes_true: np.ndarray, boxes_detection: np.ndarray) -> np.ndarray:
    # print(f"boxes true: {boxes_true}")
    # print(f"boxes detected: {boxes_detection}")
    area_true = box_area(boxes_true.T)
    area_detection = box_area(boxes_detection.T)
    # print(f"area_true: {area_true}")

    top_left = np.maximum(boxes_true[:, None, :2], boxes_detection[:, :2])
    # print(f"top_left: {top_left}")
    bottom_right = np.minimum(boxes_true[:, None, 2:4], boxes_detection[:, 2:4])
    # print(f"bottom_right: {bottom_right}")

    area_inter = np.prod(np.clip(bottom_right - top_left, a_min=0, a_max=None), 2)
    # print(f"area_inter: {area_inter}")
    return area_inter / (area_true[:, None] + area_detection - area_inter)


def iou_distance(atracks: List, btracks: List) -> np.ndarray:
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
        len(btracks) > 0 and isinstance(btracks[0], np.ndarray)
    ):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]

    _ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
    if _ious.size != 0:
        _ious = box_iou_batch(np.asarray(atlbrs), np.asarray(btlbrs))
    cost_matrix = 1 - _ious

    return cost_matrix

def box_feature_batch(pred_features, detect_features):
    # pred_features_mag = np.linalg.norm(pred_features, axis=1)
    # detect_features_mag = np.linalg.norm(detect_features, axis=1)
    # cos_sim = np.empty(pred_features.shape[0], detect_features.shape[0])
    # best_matches = np.empty(pred_features.shape[0])
    # for i in range(pred_features.shape[0]):
      # max_sim = 0
      # max_index = 0
      # cos_sim[i] = 1-np.dot(pred_features[i], detected_features)/(pred_features_mag[i]*detect_features_mag)
      # for j in range(detect_featurees.shape[0])
      #   cos_dist[i,j] = 1-np.dot(pred_features[i], detect_features[j])/(pred_features_mag[i]*detect_features_mag[j])
      #   if cos_dist[i,j] > max_sim:
      #     max_index = j
      #     max_sim = cos_dist[i,j]
      # best_matches[i] = max_index
    cos_dist = 1 - cosine_similarity(pred_features, detect_features)
    cos_dist = (cos_dist)/2 # rescale to [0,1]
    # cos_dist = cos_dist*10 # seems to have range less than 0.1 so scale it up
    return cos_dist

def feature_distance(atracks: List, btracks: List) -> np.ndarray:
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
        len(btracks) > 0 and isinstance(btracks[0], np.ndarray)
    ):
        a_features = atracks
        b_features = btracks
    else:
        a_features = [track.feature for track in atracks]
        b_features = [track.feature for track in btracks]

    cost_matrix = np.zeros((len(a_features), len(b_features)), dtype=np.float32)
    if cost_matrix.size != 0:
        cost_matrix = box_feature_batch(np.asarray(a_features), np.asarray(b_features))
        cost_matrix = (cost_matrix)/(.01)

    return cost_matrix

def fuse_bb_score(cost_matrix: np.ndarray, detections: List) -> np.ndarray:
    if cost_matrix.size == 0:
        return cost_matrix

    iou_sim = 1 - cost_matrix
    # feature_sim = 1 - feature_cost_matrix
    det_scores = np.array([det.score for det in detections])
    # print("conf scores: ", det_scores)
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost

def fuse_feature_cost(cost_matrix: np.ndarray, feature_cost_matrix: np.ndarray, feature_weight=0.5) -> np.ndarray:
    if cost_matrix.size == 0:
        return cost_matrix
    if feature_weight > 1:
        feature_weight = 1
    # # change to similarity because we want our similarity to get smaller, increasing the cost
    # cost_sim = 1 - ((1/feature_weight) * cost_matrix)
    # feature_sim = 1 - (feature_weight * feature_cost_matrix)
    # # fuse_sim = ((cost_sim) + (feature_weight * feature_sim)) / (1+feature_weight)
    # fuse_sim = cost_sim * feature_sim
    # fuse_cost = 1 - fuse_sim
    fuse_cost = (1-feature_weight) * cost_matrix + (feature_weight) * feature_cost_matrix
    return fuse_cost

# def match_tracks(predictions_bb, detections, predictions_feature, yolo_box_features):

#   dists = box_iou_batch(predictions_bb, detections)
#   feature_dists = box_feature_batch(predictions_feature, yolo_box_features)

#   dists = fuse_score(dists, feature_dists, detections)

#   matches, untracked, undetected = linear_assignment(dists, thresh=0.7)

#   for index in untracked: # possible tracks
#     box = dists[index]
#     track = Track(NEXT_ID) # Global var

#     tracked_objects.update({track.track_id:track})
#     # start feature history

#   for index in undetected:
#     tracked_objects[predictions[index].track_id].mark_lost()

def indices_to_matches(cost_matrix: np.ndarray, indices: np.ndarray, thresh: float) -> Tuple[np.ndarray, tuple, tuple]:
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = matched_cost <= thresh
    # print(f"matched mask: {matched_mask}")
    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))
    return matches, unmatched_a, unmatched_b

def linear_assignment(cost_matrix: np.ndarray, thresh: float):
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            tuple(range(cost_matrix.shape[0])),
            tuple(range(cost_matrix.shape[1])),
        )
    # print(f"cost matrix pre lin assign: {cost_matrix}")
    cost_matrix[cost_matrix > thresh] = thresh + 1e-4
    # print(f"lin assign cost matrix: {cost_matrix}")
    row_ind, col_ind = linear_sum_assignment(cost_matrix) # linear_sum_assignment from scipy optimize
    # need to change to when two boxes are very close, then I need to wait for a frame or two
    indices = np.column_stack((row_ind, col_ind))

    return indices_to_matches(cost_matrix, indices, thresh)

''' should find a better spot for this'''
def extract_features(bbox, img, feature_extractor):
  yolo_box_features = np.empty([bbox.shape[0], 512])
  for i, box in enumerate(bbox):
    box[box < 0] = 0
    # print(box)
    bb_left = int(abs(box[0]))
    bb_top = int(abs(box[1]))
    bb_right = int(abs(box[2]))
    bb_bottom = int(abs(box[3]))
    # bb_left, bb_top, bb_width, bb_height = bbox
    roi = img.copy()[bb_top:bb_bottom, bb_left:bb_right]
    # print(img)
    # print(roi)
    # resize the image to be the correct input size for the CNN feature extractor
    img_resize = cv2.resize(roi, (256, 128))
    # Normalize the image to make pixel vals scaled to [0,1]
    img_norm = cv2.normalize(img_resize, None, 0.0, 1.0, cv2.NORM_MINMAX)
    feature_vect = feature_extractor(img_norm).cpu()
    # print(feature_vect, feature_vect.shape)

    yolo_box_features[i] = feature_vect


  return yolo_box_features

class LSTM_Track:
    """
    Initialize the LSTM_Track object.

    Parameters:
        model: the LSTM prediction model to use.
        track_thresh (float, optional): Detection confidence threshold
            for track activation.
        track_buffer (int, optional): Number of frames to buffer when a track is lost.
        match_thresh (float, optional): Threshold for matching tracks with detections.
        frame_rate (int, optional): The frame rate of the video.
    """

    def __init__(
        self,
        model,
        track_thresh: float = 0.25,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        with_reid: bool = True,
        torchreid_model="./models/osnet_ms_d_c.pth.tar", # move this into folder
        frame_rate: int = 30,
    ):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh

        self.frame_id = 0
        self.det_thresh = self.track_thresh + 0.1
        # self.max_time_lost = int(frame_rate / 30.0 * track_buffer)
        self.max_time_lost = track_buffer
        self.LSTM_predictor = LSTM_predictor(model=model) # can change this

        self.tracked_tracks: List[STrack] = []
        self.lost_tracks: List[STrack] = []
        self.removed_tracks: List[STrack] = []

        self.prev_len = 0

        self.with_reid = with_reid

        ''' have to use special reid model, or re-train the LSTM model'''
        if self.with_reid:
          self.feature_extractor = ut.FeatureExtractor(
            model_name='osnet_x1_0',
            model_path=torchreid_model,
            device='cuda'
          )


    def update_with_detections(self, detections: Detections, detection_features: np.ndarray, img, resize) -> Detections:
        """
        Updates the tracker with the provided detections and
            returns the updated detection results.

        Parameters:
            detections: The new detections to update with.
            detection_features: The image features extracted by the re-id model.
            They correspond to the detections
        Returns:
            Detection: The updated detection results that now include tracking IDs.
        Example:
            ```python
            >>> import supervision as sv
            >>> from ultralytics import YOLO

            >>> model = YOLO(...)
            >>> byte_tracker = sv.ByteTrack()
            >>> annotator = sv.BoxAnnotator()

            >>> def callback(frame: np.ndarray, index: int) -> np.ndarray:
            ...     results = model(frame)[0]
            ...     detections = sv.Detections.from_ultralytics(results)
            ...     detections = byte_tracker.update_with_detections(detections)
            ...     labels = [
            ...         f"#{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
            ...         for _, _, confidence, class_id, tracker_id
            ...         in detections
            ...     ]
            ...     return annotator.annotate(scene=frame.copy(),
            ...                               detections=detections, labels=labels)

            >>> sv.process_video(
            ...     source_path='...',
            ...     target_path='...',
            ...     callback=callback
            ... )
            ```
        """
  
        tracks = self.update_with_tensors(
            # maybe extract features here
            tensors=detections2boxes(detections=detections, features=detection_features),
            img=img, res=img.shape
        ) 
        detections = Detections.empty()
        if len(tracks) > 0:
            detections.xyxy = np.array(
                [track.tlbr for track in tracks], dtype=np.float32
            )
            detections.class_id = np.array(
                [int(t.cls) for t in tracks], dtype=int
            )
            detections.tracker_id = np.array(
                [int(t.track_id) for t in tracks], dtype=int
            )
            detections.confidence = np.array(
                [t.score for t in tracks], dtype=np.float32
            )
        else:
            detections.tracker_id = np.array([], dtype=int)

        return detections

    def update_with_tensors(self, tensors: np.ndarray, img, res) -> List[STrack]:
        """
        Updates the tracker with the provided tensors and returns the updated tracks.

        Parameters:
            tensors: The new tensors to update with.
            img: The current frame for reid
            res: The resolution of the video

        Returns:
            List[STrack]: Updated tracks.
        """
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        class_ids = tensors[:, 5]
        scores = tensors[:, 4]
        bboxes = tensors[:, :4]
        features = tensors[:,6:]

        remain_inds = scores > self.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]

        features_second = features[inds_second]
        features = features[remain_inds]

        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        class_ids_keep = class_ids[remain_inds]
        class_ids_second = class_ids[inds_second]

        if len(dets) > 0:
            """Detections"""
            detections = [
                STrack(STrack.tlbr_to_tlwh(tlbr), f, s, c)
                for (tlbr, s, c, f) in zip(dets, scores_keep, class_ids_keep, features)
            ]
        else:
            detections = []

        '''Extract embeddings '''
        if self.with_reid: 
        	# i think this will work
          if img is not None:
            features_keep = extract_features(dets, img, self.feature_extractor)
          else:
            print("img is empty!")

        """ Add newly detected tracklets to tracked_stracks"""
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_tracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        """ Step 2: First association, with high score detection boxes"""
        strack_pool = joint_tracks(tracked_stracks, self.lost_tracks)
        # if len(strack_pool) > 0:
        #   print("these are the tracks we want to predict: ", strack_pool[0].sequence)
        STrack.multi_predict(strack_pool, res)
        # print(f"num of tracks: {len(strack_pool)}   num of dets: {len(detections)}")
        dists = iou_distance(strack_pool, detections)
        first_iou_dists = dists.copy()
        print("first iou cost: ", dists)
        feature_dists = feature_distance(strack_pool, detections)
        print("feature cost: ", feature_dists)
        dists = fuse_bb_score(dists, detections)
        print("fused iou and conf cost: ", dists)
        feature_dists = fuse_feature_cost(dists, feature_dists, feature_weight=0.8)
        print("fused feature and iou and conf cost: ", feature_dists)
        matches, u_track, u_detection = linear_assignment(
            feature_dists, thresh=self.match_thresh
        )
        # print("matched tracks: ", matches)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                # print("reactivate")
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        """ Step 3: Second association, with low score detection boxes"""
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            """Detections"""
            # initialize tracks
            detections_second = [
                STrack(STrack.tlbr_to_tlwh(tlbr), f, s, c)
                for (tlbr, s, c, f) in zip(dets_second, scores_second, class_ids_second, features_second)
            ]
        else:
            detections_second = []
        r_tracked_stracks = [
            strack_pool[i]
            for i in u_track
            if strack_pool[i].state == TrackState.Tracked
        ]
        dists = iou_distance(r_tracked_stracks, detections_second)
        if dists.shape[1] >= 1:
          print("frame num: ", self.frame_id)
          print("second iou cost: ", dists)
        feature_dists = feature_distance(r_tracked_stracks, detections_second)
        if dists.shape[1] >= 1:
          print("second feature cost: ", feature_dists)
        fused_dists = fuse_feature_cost(dists, feature_dists)
        if dists.shape[1] >= 1:
          print("second fused feature and iou cost: ", fused_dists)
        matches, u_track, u_detection_second = linear_assignment(
            fused_dists, thresh=0.65 # adjust this thresh
        )
        # print("second matched tracks: ", matches)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        """Deal with unconfirmed tracks, usually tracks with only one beginning frame"""
        detections = [detections[i] for i in u_detection]
        dists = iou_distance(unconfirmed, detections)
        print("unconfirmed iou cost: ", dists)
        # probably want to include feature cost in this.
        dists = fuse_bb_score(dists, detections)
        print("unconfirmed fused iou and conf cost: ", dists)
        matches, u_unconfirmed, u_detection = linear_assignment(
            dists, thresh=0.7
        )
        # print(f"unconfirmed matches: {matches}")
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            # remember that detections should be an initialized track
            track.activate(self.LSTM_predictor, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_tracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                # print(f"remove track: {track.track_id}")
                # print(f"lost time: {self.frame_id - track.end_frame}")
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_tracks = [
            t for t in self.tracked_tracks if t.state == TrackState.Tracked
        ]
        self.tracked_tracks = joint_tracks(self.tracked_tracks, activated_starcks)
        self.tracked_tracks = joint_tracks(self.tracked_tracks, refind_stracks)
        self.lost_tracks = sub_tracks(self.lost_tracks, self.tracked_tracks)
        self.lost_tracks.extend(lost_stracks)
        # print(f"lost tracks pre: {self.lost_tracks}")
        self.lost_tracks = sub_tracks(self.lost_tracks, removed_stracks)
        # print(f"lost tracks post: {self.lost_tracks}")
        # self.removed_tracks.extend(removed_stracks)
        self.tracked_tracks, self.lost_tracks = remove_duplicate_tracks(
            self.tracked_tracks, self.lost_tracks
        )
        output_stracks = [track for track in self.tracked_tracks if track.is_activated]
        # if len(output_stracks) != self.prev_len:
        #   print("frame num: ", self.frame_id)
        #   print("diff num of active tracks: ", first_iou_dists)
        #   self.prev_len = len(output_stracks)
        if self.frame_id % 10 == 0:
          print(f"size of removed_tracks: {len(self.removed_tracks)}, size of lost tracks: {len(self.lost_tracks)}, size of tracked: {len(self.tracked_tracks)}")
        return output_stracks


def joint_tracks(
    track_list_a: List[STrack], track_list_b: List[STrack]
) -> List[STrack]:
    """
    Joins two lists of tracks, ensuring that the resulting list does not
    contain tracks with duplicate track_id values.

    Parameters:
        track_list_a: First list of tracks (with track_id attribute).
        track_list_b: Second list of tracks (with track_id attribute).

    Returns:
        Combined list of tracks from track_list_a and track_list_b
            without duplicate track_id values.
    """
    seen_track_ids = set()
    result = []

    for track in track_list_a + track_list_b:
        if track.track_id not in seen_track_ids:
            seen_track_ids.add(track.track_id)
            result.append(track)

    return result


def sub_tracks(track_list_a: List, track_list_b: List) -> List[int]:
    """
    Returns a list of tracks from track_list_a after removing any tracks
    that share the same track_id with tracks in track_list_b.

    Parameters:
        track_list_a: List of tracks (with track_id attribute).
        track_list_b: List of tracks (with track_id attribute) to
            be subtracted from track_list_a.
    Returns:
        List of remaining tracks from track_list_a after subtraction.
    """
    tracks = {track.track_id: track for track in track_list_a}
    track_ids_b = {track.track_id for track in track_list_b}

    for track_id in track_ids_b:
        tracks.pop(track_id, None)

    return list(tracks.values())


def remove_duplicate_tracks(tracks_a: List, tracks_b: List) -> Tuple[List, List]:
    pairwise_distance = matching.iou_distance(tracks_a, tracks_b)
    matching_pairs = np.where(pairwise_distance < 0.15)

    duplicates_a, duplicates_b = set(), set()
    for track_index_a, track_index_b in zip(*matching_pairs):
        time_a = tracks_a[track_index_a].frame_id - tracks_a[track_index_a].start_frame
        time_b = tracks_b[track_index_b].frame_id - tracks_b[track_index_b].start_frame
        if time_a > time_b:
            duplicates_b.add(track_index_b)
        else:
            duplicates_a.add(track_index_a)

    result_a = [
        track for index, track in enumerate(tracks_a) if index not in duplicates_a
    ]
    result_b = [
        track for index, track in enumerate(tracks_b) if index not in duplicates_b
    ]

    return result_a, result_b