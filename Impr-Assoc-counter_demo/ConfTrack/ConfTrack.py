import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

from ConfTrack.matching import iou_distance, d_iou_distance, embedding_distance, fuse_motion, fuse_score, linear_assignment, ious, fuse_iou
# from tracker.gmc import GMC
from ConfTrack.basetrack import BaseTrack, TrackState
from ConfTrack.kalman_filter import KalmanFilter
from supervision.detection.core import Detections

from fast_reid.fast_reid_interfece import FastReIDInterface


class STrack(BaseTrack):
    shared_kalman = KalmanFilter(0.6, 10)

    def __init__(self, tlwh, score, class_id, feat=None, feat_history=50):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        self.cls = -1
        self.cls_hist = []  # (cls id, freq)
        self.update_cls(class_id, score)

        self.smooth_feat = None
        self.curr_feat = None
        if feat is not None:
            self.update_features(feat)
        self.features = deque([], maxlen=feat_history)
        self.alpha = 0.9

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

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
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0

        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][6] = 0
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())

                stracks[i].mean = mean
                stracks[i].covariance = cov



    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()

        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xywh(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):

        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_track.tlwh), self.score)
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

        self.update_cls(new_track.cls, new_track.score)


    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh

        self.score = new_track.score
        # print(f"{self.track_id}'s score: {self.score}")

        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_tlwh), self.score)

        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)

        self.state = TrackState.Tracked
        self.is_activated = True

        self.update_cls(new_track.cls, new_track.score)



    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def xywh(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2.0
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

    @staticmethod
    def tlwh_to_xywh(tlwh):
        """Convert bounding box to format `(center x, center y, width,
        height)`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    def to_xywh(self):
        return self.tlwh_to_xywh(self.tlwh)

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
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)

def detections2boxes(detections: Detections) -> np.ndarray:
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
            detections.class_id[:, np.newaxis]
        )
    )


class ConfTrack:
    def __init__(self,
                 track_high_thresh=0.6,
                 track_low_thresh=0.2,
                 new_track_thresh=0.1,
                 tent_conf_thresh=0.7,
                 match_thresh=0.8,
                 track_buffer=30,
                 proximity_thresh=0.6,
                 appearance_thresh=0.25,
                 with_reid=False,
                 fast_reid_config=r"fast_reid/configs/MOT17/sbs_S50.yml", #need to download
                 fast_reid_weights=r"pretrained/mot17_sbs_S50.pth", #need to download
                 device="gpu",

                 frame_rate=30):

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        BaseTrack.clear_count()

        self.frame_id = 0
        # self.args = args

        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh

        self.tent_conf_thresh = tent_conf_thresh

        self.match_thresh = match_thresh

        # self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = track_buffer
        self.kalman_filter = KalmanFilter(0.6, 10)

        # ReID module
        self.proximity_thresh = proximity_thresh
        self.appearance_thresh = appearance_thresh

        self.with_reid = with_reid

        if with_reid:
            self.encoder = FastReIDInterface(fast_reid_config, fast_reid_weights, device)

        # self.gmc = GMC(method=args.cmc_method, verbose=[args.name, args.ablation])

    def update_with_detections(self, detections: Detections, img) -> Detections:
        """
        Updates the tracker with the provided detections and
            returns the updated detection results.

        Parameters:
            detections: The new detections to update with.
            img: The image for extracting features for Re-ID
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
        tensors = detections2boxes(detections)
        # print(f"tensors: {tensors}")
        tracks = self.update_with_tensors(
            # maybe extract features here
            tensors,
            img
        )
        detections = Detections.empty()
        if len(tracks) > 0:
            detections.xyxy = np.array(
                [track.tlbr for track in tracks], dtype=float
            )
            detections.class_id = np.array(
                [int(t.cls) for t in tracks], dtype=int
            )
            detections.tracker_id = np.array(
                [int(t.track_id) for t in tracks], dtype=int
            )
            detections.confidence = np.array(
                [t.score for t in tracks], dtype=float
            )
        else:
            detections.tracker_id = np.array([], dtype=int)

        return detections

    def update_with_tensors(self, tensors, img):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if len(tensors):
            if tensors.shape[1] == 6:
                scores = tensors[:, 4]
                bboxes = tensors[:, :4]
                classes = tensors[:, -1]
                # print(f"scores: {scores}")
            else:
                scores = tensors[:, 4] * tensors[:, 5]
                bboxes = tensors[:, :4]  # x1y1x2y2
                classes = tensors[:, -1]

            # Remove bad detections
            lowest_inds = scores > self.track_low_thresh
            bboxes = bboxes[lowest_inds]
            scores = scores[lowest_inds]
            classes = classes[lowest_inds]

            # Find high threshold detections
            remain_inds = scores > self.track_high_thresh
            dets = bboxes[remain_inds]
            scores_keep = scores[remain_inds]
            classes_keep = classes[remain_inds]

            # print(f"high dets: {dets}")
        else:
            bboxes = []
            scores = []
            classes = []
            dets = []
            scores_keep = []
            classes_keep = []

        '''Extract embeddings '''
        if self.with_reid:
            features_keep = self.encoder.inference(img, dets)

        if len(dets) > 0:
            '''Detections'''
            if self.with_reid:
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, cl, f) for
                              (tlbr, s, cl, f) in zip(dets, scores_keep, classes_keep, features_keep)]
            else:
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, cl) for
                              (tlbr, s, cl) in zip(dets, scores_keep, classes_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        low_tent = []
        high_tent = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                # unconfirmed.append(track)
                # implement LM from ConfTrack paper
                if track.score < self.tent_conf_thresh:
                  low_tent.append(track)
                else:
                  high_tent.append(track)
            else:
                tracked_stracks.append(track)
        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        # print(f"high_tent: {high_tent}")
        strack_pool = joint_stracks(strack_pool, high_tent) # LM algorithm
        strack_u = joint_stracks(strack_pool, low_tent)

        # no camera motion adjustment because our device is stationary
        # # Fix camera motion
        # warp = self.gmc.apply(img, dets)
        # STrack.multi_gmc(strack_pool, warp)
        # STrack.multi_gmc(unconfirmed, warp)

        # Associate with high score detection boxes
        ious_dists = iou_distance(strack_pool, detections)
        ious_dists_mask = (ious_dists > self.proximity_thresh)

        # if not self.args.mot20:
        ious_dists = fuse_score(ious_dists, detections)

        if self.with_reid:
            emb_dists = embedding_distance(strack_pool, detections) / 2.0
            raw_emb_dists = emb_dists.copy()
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(ious_dists, emb_dists)

            # Popular ReID method (JDE / FairMOT)
            # raw_emb_dists = matching.embedding_distance(strack_pool, detections)
            # dists = matching.fuse_motion(self.kalman_filter, raw_emb_dists, strack_pool, detections)
            # emb_dists = dists

            # IoU making ReID
            # dists = matching.embedding_distance(strack_pool, detections)
            # dists[ious_dists_mask] = 1.0
        else:
            dists = ious_dists

        matches, track_conf_remain, det_high_remain = linear_assignment(dists, thresh=self.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        if len(scores):
            inds_high = scores < self.track_high_thresh
            inds_low = scores > self.track_low_thresh
            inds_second = np.logical_and(inds_low, inds_high)
            dets_second = bboxes[inds_second]
            scores_second = scores[inds_second]
            classes_second = classes[inds_second]
        else:
            dets_second = []
            scores_second = []
            classes_second = []

        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s, cl) for
                                 (tlbr, s, cl) in zip(dets_second, scores_second, classes_second)]
        else:
            detections_second = []

        r_tracked_stracks = [strack_pool[i] for i in track_conf_remain if strack_pool[i].state == TrackState.Tracked]
        dists = iou_distance(r_tracked_stracks, detections_second)
        matches, track_conf_remain, det_low_remain = linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)


        # implement LM from ConfTrack paper
        '''Step 4: low-confidence track matching with high-conf dets'''
        # Associate with high score detection boxes
        stracks_conf_remain = [r_tracked_stracks[i] for i in track_conf_remain]
        ious_dists = iou_distance(low_tent, stracks_conf_remain)
        _, low_tent_valid, _ = linear_assignment(ious_dists, thresh=1-0.7) # want to get rid of tracks with low iou costs
        stracks_low_tent_valid = [low_tent[i] for i in low_tent_valid]
        stracks_det_high_remain = [detections[i] for i in det_high_remain]
        C_low_ious = iou_distance(stracks_low_tent_valid, stracks_det_high_remain)
        ious_dists_mask = (C_low_ious > self.proximity_thresh)

        if self.with_reid:
            emb_dists = embedding_distance(stracks_low_tent_valid, stracks_det_high_remain) / 2.0
            raw_emb_dists = emb_dists.copy()
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(C_low_ious, emb_dists)
        else:
            dists = C_low_ious

        matches, track_tent_remain, det_high_remain = linear_assignment(dists, thresh=0.3) # need to find this val in ConfTrack paper

        for itracked, idet in matches:
            low_tent[itracked].update(stracks_det_high_remain[idet], self.frame_id)
            activated_starcks.append(low_tent[itracked])


        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        for it in track_tent_remain:
            track = stracks_low_tent_valid[it]
            track.mark_removed()
            removed_stracks.append(track)
        # left over confirmed tracks get lost
        for it in track_conf_remain:
            # print(f"size of stracks_conf_remain: {len(stracks_conf_remain)}")
            # print(f"index: {it}")
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # detections = [detections[i] for i in u_detection]
        # ious_dists = iou_distance(unconfirmed, detections)
        # ious_dists_mask = (ious_dists > self.proximity_thresh)
        # if not self.args.mot20:
        #     ious_dists = fuse_score(ious_dists, detections)

        # if self.args.with_reid:
        #     emb_dists = embedding_distance(unconfirmed, detections) / 2.0
        #     raw_emb_dists = emb_dists.copy()
        #     emb_dists[emb_dists > self.appearance_thresh] = 1.0
        #     emb_dists[ious_dists_mask] = 1.0
        #     dists = np.minimum(ious_dists, emb_dists)
        # else:
        #     dists = ious_dists

        # matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=0.7)
        # for itracked, idet in matches:
        #     unconfirmed[itracked].update(detections[idet], self.frame_id)
        #     activated_starcks.append(unconfirmed[itracked])



        """ Step 5: Init new stracks"""
        # u_detection = [*det_high_remain, *det_low_remain]
        for inew in det_high_remain:
            track = stracks_det_high_remain[inew]
            # print(f"init new track {track.track_id} with score :{track.score}")
            if track.score < self.new_track_thresh:
                continue

            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        for inew in det_low_remain:
            track = detections_second[inew]
            if track.score < self.new_track_thresh:
                continue

            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        """ Step 6: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        """ Merge """
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        # output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        output_stracks = [track for track in self.tracked_stracks]


        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb