import unittest

from webcam_effect.state import PosePrediction, PoseStateMachine


class PoseStateMachineTest(unittest.TestCase):
    def test_activates_when_smoothed_kicau_confidence_crosses_threshold(self):
        state = PoseStateMachine(activate_threshold=0.7, deactivate_threshold=0.4)

        active = state.update(
            [
                PosePrediction(label="kicau", confidence=0.8),
                PosePrediction(label="kicau", confidence=0.7),
                PosePrediction(label="none", confidence=0.2),
            ]
        )

        self.assertTrue(active)

    def test_stays_active_until_confidence_falls_below_deactivate_threshold(self):
        state = PoseStateMachine(activate_threshold=0.7, deactivate_threshold=0.4)
        state.update([PosePrediction(label="kicau", confidence=0.9)] * 3)

        active = state.update(
            [
                PosePrediction(label="kicau", confidence=0.5),
                PosePrediction(label="none", confidence=0.6),
                PosePrediction(label="kicau", confidence=0.5),
            ]
        )

        self.assertTrue(active)

        active = state.update([PosePrediction(label="none", confidence=0.9)] * 3)

        self.assertFalse(active)


if __name__ == "__main__":
    unittest.main()
