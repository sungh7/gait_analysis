"""
Advanced Label Swap Detection for V5.4

Key innovation: Multiple detection strategies combined
1. GT-based cross-matching (existing)
2. Bilateral symmetry assumption (NEW)
3. Prediction confidence scoring (NEW)
4. Ensemble voting (NEW)

For subjects like S1_27, S1_11, S1_16 where:
- Left prediction is nearly perfect
- Right prediction is terrible
- GT shows bilateral symmetry
→ High confidence label swap!
"""

import numpy as np
from typing import Dict, Tuple, List


class AdvancedLabelSwapDetector:
    """
    Multi-strategy label swap detection.

    Strategies:
    1. GT cross-matching (original)
    2. Bilateral symmetry violation
    3. Unilateral accuracy disparity
    4. Ensemble voting
    """

    def __init__(self,
                 gt_threshold: float = 0.95,
                 symmetry_threshold: float = 0.15,
                 disparity_threshold: float = 3.0):
        """
        Args:
            gt_threshold: GT cross-matching threshold (0.95 = 5% improvement)
            symmetry_threshold: GT bilateral symmetry threshold (15% difference)
            disparity_threshold: Left/right accuracy disparity threshold (3x worse)
        """
        self.gt_threshold = gt_threshold
        self.symmetry_threshold = symmetry_threshold
        self.disparity_threshold = disparity_threshold

    def detect_label_swap_advanced(
        self,
        left_pred: float,
        right_pred: float,
        left_gt: float,
        right_gt: float
    ) -> Dict:
        """
        Advanced multi-strategy label swap detection.

        Returns:
            {
                'swap_needed': bool,
                'confidence': float (0-100),
                'strategies': {
                    'gt_cross_matching': {...},
                    'bilateral_symmetry': {...},
                    'accuracy_disparity': {...}
                },
                'vote': str (SWAP or KEEP)
            }
        """

        # Strategy 1: GT Cross-Matching (existing)
        gt_result = self._gt_cross_matching(left_pred, right_pred, left_gt, right_gt)

        # Strategy 2: Bilateral Symmetry Check
        symmetry_result = self._bilateral_symmetry_check(left_pred, right_pred, left_gt, right_gt)

        # Strategy 3: Accuracy Disparity Analysis
        disparity_result = self._accuracy_disparity_check(left_pred, right_pred, left_gt, right_gt)

        # Ensemble Voting
        votes = {
            'gt_cross_matching': gt_result['vote'],
            'bilateral_symmetry': symmetry_result['vote'],
            'accuracy_disparity': disparity_result['vote']
        }

        swap_votes = sum(1 for v in votes.values() if v == 'SWAP')
        total_votes = len(votes)

        # Decision: Majority vote
        swap_needed = swap_votes >= 2  # At least 2/3 strategies

        # Confidence: Weighted average
        confidences = [
            gt_result['confidence'] * 0.4,      # 40% weight (most reliable)
            symmetry_result['confidence'] * 0.3, # 30% weight
            disparity_result['confidence'] * 0.3 # 30% weight
        ]

        overall_confidence = sum(confidences)

        return {
            'swap_needed': swap_needed,
            'confidence': overall_confidence,
            'vote_ratio': f"{swap_votes}/{total_votes}",
            'strategies': {
                'gt_cross_matching': gt_result,
                'bilateral_symmetry': symmetry_result,
                'accuracy_disparity': disparity_result
            }
        }

    def _gt_cross_matching(self, left_pred, right_pred, left_gt, right_gt) -> Dict:
        """Strategy 1: GT-based cross-matching (original method)"""

        # Normal matching errors
        normal_left_err = abs(left_pred - left_gt)
        normal_right_err = abs(right_pred - right_gt)
        normal_score = max(normal_left_err, normal_right_err)

        # Cross matching errors
        cross_left_err = abs(left_pred - right_gt)
        cross_right_err = abs(right_pred - left_gt)
        cross_score = max(cross_left_err, cross_right_err)

        # Decision
        swap_needed = cross_score < normal_score * self.gt_threshold

        if normal_score > 0:
            confidence = abs(normal_score - cross_score) / normal_score * 100
        else:
            confidence = 0

        vote = 'SWAP' if swap_needed else 'KEEP'

        return {
            'vote': vote,
            'confidence': confidence,
            'normal_score': normal_score,
            'cross_score': cross_score,
            'improvement': (normal_score - cross_score) / max(normal_score, 0.001) * 100
        }

    def _bilateral_symmetry_check(self, left_pred, right_pred, left_gt, right_gt) -> Dict:
        """
        Strategy 2: Bilateral symmetry assumption.

        Healthy gait GT should show bilateral symmetry (left ≈ right).
        If GT is symmetric but predictions are very asymmetric,
        check if swapping would improve symmetry.
        """

        # GT symmetry
        gt_avg = (left_gt + right_gt) / 2
        gt_symmetry = abs(left_gt - right_gt) / gt_avg

        # Prediction symmetry (current)
        pred_avg = (left_pred + right_pred) / 2
        pred_symmetry = abs(left_pred - right_pred) / pred_avg

        # Prediction symmetry (after swap)
        swap_symmetry = abs(left_pred - left_gt) / left_gt + abs(right_pred - right_gt) / right_gt
        swap_symmetry /= 2  # Average

        current_symmetry = abs(left_pred - right_pred) / pred_avg

        # If GT is symmetric (<15% difference)
        if gt_symmetry < self.symmetry_threshold:
            # Check if current prediction violates symmetry badly
            if pred_symmetry > 0.3:  # >30% asymmetry is suspicious
                # Would swapping improve?
                # After swap: left_pred would match right_gt, right_pred would match left_gt
                swap_left_err = abs(left_pred - right_gt)
                swap_right_err = abs(right_pred - left_gt)
                swap_avg_err = (swap_left_err + swap_right_err) / 2

                normal_left_err = abs(left_pred - left_gt)
                normal_right_err = abs(right_pred - right_gt)
                normal_avg_err = (normal_left_err + normal_right_err) / 2

                if swap_avg_err < normal_avg_err * 0.8:
                    confidence = (1 - swap_avg_err / max(normal_avg_err, 0.001)) * 100
                    vote = 'SWAP'
                else:
                    confidence = (1 - normal_avg_err / max(swap_avg_err, 0.001)) * 100
                    vote = 'KEEP'
            else:
                # Predictions are already symmetric
                confidence = 0
                vote = 'KEEP'
        else:
            # GT itself is asymmetric (e.g., pathological gait)
            confidence = 0
            vote = 'ABSTAIN'

        return {
            'vote': vote,
            'confidence': confidence,
            'gt_symmetry': gt_symmetry * 100,  # Percent
            'pred_symmetry': pred_symmetry * 100
        }

    def _accuracy_disparity_check(self, left_pred, right_pred, left_gt, right_gt) -> Dict:
        """
        Strategy 3: Unilateral accuracy disparity.

        If one side is nearly perfect (<5% error) while the other is terrible (>20% error),
        this suggests label swap.

        Example: S1_27
        - Left: 1.42cm error (2.2%) ✓ Perfect!
        - Right: 39.28cm error (58%) ✗ Terrible!
        → Likely labels are swapped
        """

        left_err = abs(left_pred - left_gt)
        right_err = abs(right_pred - right_gt)

        left_err_pct = left_err / left_gt * 100
        right_err_pct = right_err / right_gt * 100

        # Check for extreme disparity
        if left_err_pct < 10 and right_err_pct > 20:
            # Left is good, right is bad → Check if swap helps
            swap_left_err_pct = abs(left_pred - right_gt) / right_gt * 100
            swap_right_err_pct = abs(right_pred - left_gt) / left_gt * 100

            if swap_left_err_pct < 20 and swap_right_err_pct < 10:
                # After swap: both sides become reasonable
                disparity_ratio = right_err / max(left_err, 0.001)
                confidence = min(disparity_ratio / self.disparity_threshold * 100, 100)
                vote = 'SWAP'
            else:
                confidence = 0
                vote = 'KEEP'

        elif right_err_pct < 10 and left_err_pct > 20:
            # Right is good, left is bad → Also suspicious
            swap_left_err_pct = abs(left_pred - right_gt) / right_gt * 100
            swap_right_err_pct = abs(right_pred - left_gt) / left_gt * 100

            if swap_right_err_pct < 20 and swap_left_err_pct < 10:
                disparity_ratio = left_err / max(right_err, 0.001)
                confidence = min(disparity_ratio / self.disparity_threshold * 100, 100)
                vote = 'SWAP'
            else:
                confidence = 0
                vote = 'KEEP'
        else:
            # Both sides have similar error levels
            confidence = 0
            vote = 'KEEP'

        return {
            'vote': vote,
            'confidence': confidence,
            'left_error_pct': left_err_pct,
            'right_error_pct': right_err_pct,
            'disparity_ratio': max(left_err, right_err) / (min(left_err, right_err) + 0.001)
        }


def test_detector_on_problems():
    """Test the advanced detector on known problem subjects"""

    detector = AdvancedLabelSwapDetector(
        gt_threshold=0.95,
        symmetry_threshold=0.15,
        disparity_threshold=3.0
    )

    # Test cases from problematic subjects
    test_cases = [
        {
            'subject': 'S1_27',
            'left_pred': 66.66, 'right_pred': 106.82,
            'left_gt': 65.24, 'right_gt': 67.55,
            'expected': True  # Should detect swap
        },
        {
            'subject': 'S1_11',
            'left_pred': 58.59, 'right_pred': 87.20,
            'left_gt': 57.77, 'right_gt': 57.39,
            'expected': True  # Should detect swap
        },
        {
            'subject': 'S1_16',
            'left_pred': 61.86, 'right_pred': 42.84,
            'left_gt': 61.87, 'right_gt': 62.96,
            'expected': True  # Should detect swap
        },
        {
            'subject': 'S1_10',
            'left_pred': 70.60, 'right_pred': 70.60,
            'left_gt': 70.72, 'right_gt': 70.66,
            'expected': False  # Should NOT swap (already perfect)
        }
    ]

    print("="*90)
    print("ADVANCED LABEL SWAP DETECTOR - TEST RESULTS")
    print("="*90)

    correct = 0
    for test in test_cases:
        result = detector.detect_label_swap_advanced(
            test['left_pred'], test['right_pred'],
            test['left_gt'], test['right_gt']
        )

        detected = result['swap_needed']
        expected = test['expected']
        is_correct = detected == expected
        correct += is_correct

        print(f"\n{test['subject']:10s} - {'✓ CORRECT' if is_correct else '✗ WRONG'}")
        print(f"  Predicted: {'SWAP' if detected else 'KEEP'} (Expected: {'SWAP' if expected else 'KEEP'})")
        print(f"  Confidence: {result['confidence']:.1f}%")
        print(f"  Vote: {result['vote_ratio']}")

        for strategy, details in result['strategies'].items():
            print(f"    {strategy:25s}: {details['vote']:8s} (conf={details['confidence']:.1f}%)")

    print(f"\n{'='*90}")
    print(f"Accuracy: {correct}/{len(test_cases)} ({correct/len(test_cases)*100:.0f}%)")
    print(f"{'='*90}")


if __name__ == "__main__":
    test_detector_on_problems()
