import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from session.device_setup_dialog import DeviceSetupDialog


_APP = None


def _application() -> QApplication:
    global _APP
    _APP = QApplication.instance() or QApplication([])
    return _APP


class DeviceSetupDialogTests(unittest.TestCase):
    def test_language_defaults_to_german_and_is_returned_in_config(self):
        _application()
        dialog = DeviceSetupDialog(populate_devices=False)

        self.assertEqual(dialog._language_combo.currentText(), "German")

        dialog._language_combo.setCurrentText("French")
        config = dialog.get_config()

        self.assertEqual(config["output_language"], "French")
        dialog.close()


if __name__ == "__main__":
    unittest.main()
