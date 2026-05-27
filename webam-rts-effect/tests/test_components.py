import unittest

from webcam_effect.components import ComponentSettings, format_components, parse_components, toggle_component


class ComponentsTest(unittest.TestCase):
    def test_parse_components_enables_selected_names(self):
        settings = parse_components("segment,hand_track")

        self.assertTrue(settings.segment)
        self.assertFalse(settings.classify)
        self.assertTrue(settings.hand_track)

    def test_parse_components_rejects_unknown_name(self):
        with self.assertRaises(ValueError):
            parse_components("segment,unknown")

    def test_format_components_preserves_order(self):
        settings = ComponentSettings(segment=False, classify=True, hand_track=True)

        self.assertEqual(format_components(settings), "classify,hand_track")

    def test_toggle_component_flips_one_value(self):
        settings = toggle_component(ComponentSettings(), "classify")

        self.assertTrue(settings.segment)
        self.assertFalse(settings.classify)
        self.assertTrue(settings.hand_track)


if __name__ == "__main__":
    unittest.main()
