import numpy as np
from models.utils import torch2numpy

#  ColorHandPose3DNetwork - Network for estimating 3D Hand Pose from a single RGB Image
#  Copyright (C) 2017  Christian Zimmermann
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.


class ZimEval:
    """ Util class for evaluation networks.
    """

    def __init__(self, num_kp=21):
        # init empty data storage
        self.data = list()
        self.num_kp = num_kp
        for _ in range(num_kp):
            self.data.append(list())

    def feed(self, keypoint_gt, keypoint_pred, keypoint_vis=None):
        """
        Used to feed data to the class.
        Stores the euclidean distance between gt and pred, when it is visible.
        """
        
        keypoint_gt = np.squeeze(torch2numpy(keypoint_gt))
        keypoint_pred = np.squeeze(torch2numpy(keypoint_pred))

        if keypoint_vis is None:
            keypoint_vis = np.ones_like(keypoint_gt[:, 0])
        keypoint_vis = np.squeeze(keypoint_vis).astype("bool")

        assert len(keypoint_gt.shape) == 2
        assert len(keypoint_pred.shape) == 2
        assert len(keypoint_vis.shape) == 1

        # calc euclidean distance
        diff = keypoint_gt - keypoint_pred
        euclidean_dist = np.sqrt(np.sum(np.square(diff), axis=1))

        num_kp = keypoint_gt.shape[0]
        for i in range(num_kp):
            if keypoint_vis[i]:
                self.data[i].append(euclidean_dist[i])
        return np.mean(euclidean_dist)

    def _get_pck(self, kp_id, threshold):
        """ Returns pck for one keypoint for the given threshold. """
        if len(self.data[kp_id]) == 0:
            return None

        data = np.array(self.data[kp_id])
        pck = np.mean((data <= threshold).astype("float"))
        return pck

    def _get_epe(self, kp_id):
        """ Returns end point error for one keypoint. """
        if len(self.data[kp_id]) == 0:
            return None, None

        data = np.array(self.data[kp_id])
        epe_mean = np.mean(data)
        epe_median = np.median(data)
        return epe_mean, epe_median

    def get_measures(self, val_min, val_max, steps):
        """ Outputs the average mean and median error as well as the pck score. """
        thresholds = np.linspace(val_min, val_max, steps)
        thresholds = np.array(thresholds)
        norm_factor = np.trapz(np.ones_like(thresholds), thresholds)

        # init mean measures
        epe_mean_all = list()
        epe_median_all = list()
        auc_all = list()
        pck_curve_all = list()

        # Create one plot for each part
        for part_id in range(self.num_kp):
            # mean/median error
            mean, median = self._get_epe(part_id)

            if mean is None:
                # there was no valid measurement for this keypoint
                continue

            epe_mean_all.append(mean)
            epe_median_all.append(median)

            # pck/auc
            pck_curve = list()
            for t in thresholds:
                pck = self._get_pck(part_id, t)
                pck_curve.append(pck)

            pck_curve = np.array(pck_curve)
            pck_curve_all.append(pck_curve)
            auc = np.trapz(pck_curve, thresholds)
            auc /= norm_factor
            auc_all.append(auc)
        # Display error per keypoint
        epe_mean_joint = epe_mean_all
        epe_mean_all = np.mean(np.array(epe_mean_all))
        epe_median_all = np.mean(np.array(epe_median_all))
        auc_all = np.mean(np.array(auc_all))
        pck_curve_all = np.mean(
            np.array(pck_curve_all), 0
        )  # mean only over keypoints

        return (
            epe_mean_all,
            epe_mean_joint,
            epe_median_all,
            auc_all,
            pck_curve_all,
            thresholds,
        )


def feed_evaluators_hand(evaluators, pred, gt, weights, tag, center_idx): 
    pred_joints3d = torch2numpy(pred)
    gt_joints3d=torch2numpy(gt)
    weights=torch2numpy(weights)

    if center_idx is not None:
        assert center_idx==0
        gt_joints3d_cent = gt_joints3d - gt_joints3d[:, center_idx : center_idx + 1]
        pred_joints3d_cent = pred_joints3d - pred_joints3d[:, center_idx : center_idx + 1]
        
        for cid in range(0,pred_joints3d.shape[0]):
            gt_joints,pred_joints=gt_joints3d_cent[cid],pred_joints3d_cent[cid]
            if weights[cid]>1e-4:
                evaluators[f"{tag}joints3d_cent"].feed(gt_joints[1:,], pred_joints[1:,])
    

    for cid in range(0,pred_joints3d.shape[0]):
        gt_joints,pred_joints=gt_joints3d[cid],pred_joints3d[cid]
        if weights[cid]>1e-4:
            evaluators[f"{tag}joints3d"].feed(gt_joints, pred_joints)






def parse_evaluators(evaluators, config=None):
    """
    Parse evaluators for which PCK curves and other statistics
    must be computed
    """
    if config is None:
        config = {"joints3d": [0, 0.08, 33],
            "left_joints3d": [0, 0.09, 37],
            "right_joints3d": [0, 0.09, 37],
            
            "left_joints3d_cent": [0, 0.05, 21],
            "right_joints3d_cent": [0, 0.05, 21],
            "joints3d_cent": [0, 0.05, 21],}
    eval_results = {}
    for evaluator_name, evaluator in evaluators.items():
        start, end, steps = [config[evaluator_name][idx] for idx in range(3)]
        (epe_mean, epe_mean_joints, epe_median, auc, pck_curve, thresholds) = evaluator.get_measures(start, end, steps)
        eval_results[evaluator_name] = {
            "epe_mean": epe_mean,
            "epe_mean_joints": epe_mean_joints,
            "epe_median": epe_median,
            "auc": auc,
            "thresholds": thresholds,
            "pck_curve": pck_curve,
        }
        
    return eval_results
