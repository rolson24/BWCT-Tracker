from typing import Tuple

import numpy as np
import scipy.linalg

import tensorflow as tf


class LSTM_predictor:
    """
    An LSTM predictor to predict the next state in a sequence.

    The 516-dimensional state space

        512 features in feature vect, x, y, w, h

    contains the bounding box feature vector from the feature extractor,
    top left position (x, y), width w, and height h.

    Object motion is predicted by the LSTM. The bounding box location
    (x, y, w, h) is predicted and the feature vector is also predicted.
    """

    def __init__(self, model, pred_features=True, num_features=512, n_state_dim=4, seq_len=40):
        self.model = model
        self.pred_features = pred_features
        self.ndim = n_state_dim
        self.num_features = num_features
        self.max_seq_len = seq_len
        self.tot_vect_len = num_features + n_state_dim
        # self.res = res


    def initiate(self, bb_measurement: np.ndarray, feature_vect: np.ndarray) -> tf.Tensor:
        """
        Create track from an unassociated measurement.

        Args:
            bb_measurement (ndarray): Bounding box coordinates (x, y, w, h) with
                top left position (x, y), width w, and height h.
            feature_vect (ndarray): Bounding box feature vector from the feature
                extractor model.

        Returns:
            tensor: Returns the first state tensor (feature_vect + bb_measurement)
                in the sequence.
        """
        return tf.convert_to_tensor(np.concat(feature_vect, bb_measurement))

    def predict(self, sequence: np.ndarray, tracklet_len: int, res: tuple) -> tf.Tensor:
        """
        Run LSTM prediction step.

        Args:
            sequence (ndarray): The sequence of the states of the track so far.
                Has shape [seq_len, num_features+ndim]. If length of sequence
                is less than max_seq_len, zeros will be padded to the front.
            tracklet_len (int): The length of the sequence tracked so far. Is 1
                indexed
            res (tuple): The resolution of the current frame

        Returns:
            tensor: Returns the next feature vector predicted in the sequence.
            Has shape [num_features+ndim]
        """

        # pad the sequence to make it the correct length
        # I think I don't actually need to pad these
        # if sequence.shape[0] < self.max_seq_len:
        #   paddings = tf.constant([[self.max_seq_len-sequence.shape[0], 0], [0,0]])
        #   sequence = tf.pad(sequence, paddings, "CONSTANT")
        # convert to [0,1] because thats what the predictor was trained on
        sequence[:, -4] = sequence[:, -4] / res[0]
        sequence[:, -2] = sequence[:, -2] / res[0]
        sequence[:, -3] = sequence[:, -3] / res[1]
        sequence[:, -1] = sequence[:, -1] / res[1]
        # print("sequence before predict", sequence)
        next_state = self.model.predict(tf.convert_to_tensor(sequence[:tracklet_len]), verbose=0)

        # if the model outputs a whole sequence then just pick the last value
        if next_state.shape == [self.max_seq_len, self.tot_vect_len]:
          # convert back to normal res
          next_state[:,-4] = next_state[:,-4] * res[0]
          next_state[:,-2] = next_state[:,-2] * res[0]
          next_state[:,-3] = next_state[:,-3] * res[1]
          next_state[:,-1] = next_state[:,-1] * res[1]
          return next_state[-1]
        else:
          next_state[-4] = next_state[-4] * res[0]
          next_state[-2] = next_state[-2] * res[0]
          next_state[-3] = next_state[-3] * res[1]
          next_state[-1] = next_state[-1] * res[1]
          return next_state

    def multi_predict(self, sequences: np.ndarray, track_lens: np.ndarray, res: tuple) -> tf.Tensor:
        """
        Run LSTM prediction step (Vectorized version).

        Args:
            sequences (ndarray): The Nx(max_seq_len)x(feature_len+ndim) array of
                all of the sequences from the previous tracks
            track_len (ndarray): Array of track lengths associated with each
                sequence. Has shape [N]
            res (tuple): The resolution of the current frame

        Returns:
            tensor: Returns the next feature vector predicted in all of the
                sequences. Has shape [N, num_features+ndim]
        """

        # pad the sequences to make them all the correct length
        # pad them so that the sequence is at the end. plan to add to sequence
        # from the start and go longer as the sequence gets longer until
        # tracklet_len = max_seq_len, then shift tensor with new pred on the end
        for i in range(sequences.shape[0]):
          if track_lens[i] < self.max_seq_len:
            # print("before padding: ", sequences[i])
            paddings = tf.constant([[self.max_seq_len-track_lens[i], 0], [0,0]])
            # print("pad tensor: ", paddings)
            sequences[i] = tf.pad(sequences[i, :track_lens[i]], paddings, "CONSTANT")
            # print("after padding: ", sequences[i])
          # convert to [0,1] because thats what the predictor was trained on
          sequences[i, :, -4] = sequences[i, :, -4] / res[0]
          sequences[i, :, -2] = sequences[i, :, -2] / res[0]
          sequences[i, :, -3] = sequences[i, :, -3] / res[1]
          sequences[i, :, -1] = sequences[i, :, -1] / res[1]

        # print("sequences before multi_predict", sequences)
        next_states = self.model.predict(tf.convert_to_tensor(sequences), verbose=0)

        # if the model outputs a whole sequence then just pick the last value
        if len(next_states.shape) > 2:
          # convert back to original res
          next_states[:,:,-4] = next_states[:,:,-4] * res[0]
          next_states[:,:,-2] = next_states[:,:,-2] * res[0]
          next_states[:,:,-3] = next_states[:,:,-3] * res[1]
          next_states[:,:,-1] = next_states[:,:,-1] * res[1]
          return next_states[:,-1]
        else:
          next_states[:,-4] = next_states[:,-4] * res[0]
          next_states[:,-2] = next_states[:,-2] * res[0]
          next_states[:,-3] = next_states[:,-3] * res[1]
          next_states[:,-1] = next_states[:,-1] * res[1]
          return next_states

          