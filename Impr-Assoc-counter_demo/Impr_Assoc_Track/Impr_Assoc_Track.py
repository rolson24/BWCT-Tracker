import numpy as np
from collections import deque

from Impr_Assoc_Track.matching import iou_distance, d_iou_distance, embedding_distance, fuse_motion, fuse_score, linear_assignment, ious, fuse_iou
# from tracker.gmc import GMC
from Impr_Assoc_Track.basetrack import BaseTrack, TrackState
from Impr_Assoc_Track.kalman_filter import KalmanFilter
from supervision.detection.core import Detections

from fast_reid.fast_reid_interfece import FastReIDInterface


class STrack(BaseTrack):
    shared_kalman = KalmanFilter(0.6, 10)

    def __init__(self, tlwh, score, class_id, feat=None, feat_history=15):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        # hold on to start location for counting
        self.start_tlwh = self._tlwh

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
        # if frame_id == 1:
        # from OAI track, no unconfirmed tracks.
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


class ImprAssocTrack:
    def __init__(self,
                 track_high_thresh=0.6,
                 track_low_thresh=0.1,
                 new_track_thresh=0.7,
                 match_thresh=0.65, # bigger?
                 second_match_thresh=0.19,
                 overlap_thresh=0.55,
                 iou_weight=0.2,
                 track_buffer=35,
                 proximity_thresh=0.1,
                 appearance_thresh=0.25,
                 with_reid=True,
                 fast_reid_config=r"/content/drive/MyDrive/YOLO_detections/ReID/sbs_S50.yml", #need to download
                 fast_reid_weights=r"/content/drive/MyDrive/YOLO_detections/ReID/mot17_sbs_S50.pth", #need to download
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

        # self.tent_conf_thresh = tent_conf_thresh

        self.match_thresh = match_thresh
        self.second_match_thresh = second_match_thresh

        self.overlap_thresh = overlap_thresh

        self.iou_weight = iou_weight

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

    def update_with_tensors(self, tensors, img):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if len(tensors):
            if tensors.shape[1] == 6: # need to validate
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
        # print(f"first bboxes: {bboxes}")
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
                unconfirmed.append(track)
                # # implement LM from ConfTrack paper
                # if track.score < self.tent_conf_thresh:
                #   low_tent.append(track)
                # else:
                #   high_tent.append(track)
            else:
                tracked_stracks.append(track)

        '''Improved Association: First they calc the cost matrix of the high
        detections(func_1 -> cost_h), then the calc the cost matrix of the low
        detections (func_2 -> cost_l) and get the max values of both. Then
        B = det_h_max / det_l_max.
        Finally they calc cost = concat(cost_h, B*cost_l) for the matching
        '''

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # print(f"strack pool: {strack_pool}")
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        # print(f"high_tent: {high_tent}")

        # from ConfTrack
        # strack_pool = joint_stracks(strack_pool, high_tent) # LM algorithm
        # strack_u = joint_stracks(strack_pool, low_tent)

        # no camera motion adjustment because our device is stationary
        # # Fix camera motion
        # warp = self.gmc.apply(img, dets)
        # STrack.multi_gmc(strack_pool, warp)
        # STrack.multi_gmc(unconfirmed, warp)

        # Associate with high score detection boxes
        d_ious_dists = d_iou_distance(strack_pool, detections)
        ious = 1 - iou_distance(strack_pool, detections)
        # print(f"first dets: {detections}")
        # print(f"ious_dists: {ious_dists}")
        ious_mask = (ious < self.proximity_thresh) # o_min in ImprAssoc paper

        # if not self.args.mot20:
        # ious_dists = fuse_score(ious_dists, detections)

        '''ignore this for now. Actually i think i need to integrate this otherwise
        improved association paper isn't going to be very good. Main innovation
        is combined matching, but with no Re-ID the distance functions are the
        same.'''

        if self.with_reid:
            # ConfTrack version
            # emb_dists = embedding_distance(strack_pool, detections) / 2.0
            # raw_emb_dists = emb_dists.copy()
            # emb_dists[emb_dists > self.appearance_thresh] = 1.0
            # emb_dists[ious_dists_mask] = 1.0
            # dists = np.minimum(ious_dists, emb_dists)

            # Popular ReID method (JDE / FairMOT)
            # raw_emb_dists = matching.embedding_distance(strack_pool, detections)
            # dists = matching.fuse_motion(self.kalman_filter, raw_emb_dists, strack_pool, detections)
            # emb_dists = dists

            # IoU making ReID
            # dists = matching.embedding_distance(strack_pool, detections)
            # dists[ious_dists_mask] = 1.0

            # Improved Association Version (CD)
            emb_dists = embedding_distance(strack_pool, detections) # high dets
            dists = self.iou_weight*d_ious_dists + (1-self.iou_weight)*emb_dists
            dists[ious_mask] = self.match_thresh + 0.00001
        else:
            dists = d_ious_dists
            dists[ious_mask] = self.match_thresh + 0.00001
        # print(f"dist first: {dists}, {len(dists)}")

        # Associate with low score detection boxes
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
        # print(f"dets_second: {detections_second}")
        dists_second = iou_distance(strack_pool, detections_second)
        dists_second_mask = (dists_second > self.second_match_thresh) # this is what the paper used
        dists_second[dists_second_mask] = self.second_match_thresh + 0.00001


        B = self.match_thresh/self.second_match_thresh
        # print(f"B: {B}")

        combined_dists = np.concatenate((dists, B*dists_second), axis=1) # need to double check
        # print(f"combined_dists: {combined_dists}")

        matches, track_conf_remain, det_remain = linear_assignment(combined_dists, thresh=self.match_thresh)

        # concat detections so that it all works
        detections = np.concatenate((detections, detections_second), axis=0) # double check
        # print(f"combined_dets: {detections}")
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        '''Deal with lost tracks'''

        # left over confirmed tracks get lost
        for it in track_conf_remain:
            # print(f"size of stracks_conf_remain: {len(stracks_conf_remain)}")
            # print(f"index: {it}")
            track = strack_pool[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''now do OAI from improved assoc'''
        # calc the iou between every unmatched det and all tracks if the max iou
        # for a det D is above overlap_thresh, discard it.
        # print(f"det remain: {det_remain}")
        sdet_remain = [detections[i] for i in det_remain]
        bboxes = [track.tlbr for track in sdet_remain]
        bboxes = np.array(bboxes)
        # print(f"bboxes: {bboxes}")
        if self.with_reid:
            features = self.encoder.inference(img, bboxes)

        unmatched_overlap = 1 - iou_distance(strack_pool, sdet_remain)

        for det_ind in range(unmatched_overlap.shape[1]): # loop over the rows
            if len(unmatched_overlap[:, det_ind]) != 0:
                if np.max(unmatched_overlap[:, det_ind]) < self.overlap_thresh:
                    # now initialize it
                    track = sdet_remain[det_ind]
                    # print(f"init track {track.track_id}, conf: {track.score}")
                    if track.score > self.new_track_thresh:
                        track.activate(self.kalman_filter, self.frame_id)
                        if self.with_reid:
                            track.update_features(features[det_ind])
                        activated_starcks.append(track)
            else:
                # if no curr tracks, then init one
                track = sdet_remain[det_ind]
                # print(f"init track {track.track_id}, conf: {track.score}")
                if track.score > self.new_track_thresh:
                    track.activate(self.kalman_filter, self.frame_id)
                    if self.with_reid:
                        track.update_features(features[det_ind])
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
