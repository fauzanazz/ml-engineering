import unittest

from webcam_effect.config import RuntimeConfig, apply_runtime_config, update_runtime_config
from webcam_effect.state import PoseStateMachine


class RuntimeConfigTest(unittest.TestCase):
    def test_hotkeys_update_threshold_scale_and_debug(self):
        config = RuntimeConfig()

        config = update_runtime_config(config, ord("["))
        config = update_runtime_config(config, ord("="))
        config = update_runtime_config(config, ord("d"))

        self.assertEqual(config.activate_threshold, 0.6499999999999999)
        self.assertEqual(config.sticker_scale, 0.3)
        self.assertTrue(config.debug)

    def test_applies_thresholds_to_state_machine(self):
        state = PoseStateMachine()

        apply_runtime_config(state, RuntimeConfig(activate_threshold=0.6, deactivate_threshold=0.2))

        self.assertEqual(state.activate_threshold, 0.6)
        self.assertEqual(state.deactivate_threshold, 0.2)


if __name__ == "__main__":
    unittest.main()
