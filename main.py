# main.py
import sys
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from PySide6.QtWidgets import QApplication, QMessageBox
from session.device_setup_dialog import DeviceSetupDialog
from session.sidebar_panel import SidebarPanel
from settings import Settings


def setup_logging():
    log_dir = Path.home() / "DocuFlow" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "docuflow.log"

    handler = RotatingFileHandler(log_file, maxBytes=2_000_000, backupCount=3, encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)

    # Also log to console for development visibility
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)


def check_consent(settings: Settings) -> bool:
    if settings.consent_given:
        return True

    msg = QMessageBox()
    msg.setIcon(QMessageBox.Icon.Warning)
    msg.setWindowTitle("Privacy & Data Policy Consent")
    msg.setText(
        "DocuFlow records audio locally on your machine.\n\n"
        "To generate documentation, your transcribed text and captured screenshots "
        "will be sent to a cloud LLM/VLM service (via TLS 1.3 encrypted connection).\n\n"
        "Do you consent to this data usage?"
    )
    msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
    msg.setDefaultButton(QMessageBox.StandardButton.No)

    if msg.exec() == QMessageBox.StandardButton.Yes:
        settings.consent_given = True
        settings.save()
        return True
    return False


def main():
    setup_logging()
    app = QApplication(sys.argv)
    app.setApplicationName("DocuFlow")
    app.setStyle("Fusion")

    settings = Settings.load()

    if not check_consent(settings):
        sys.exit(0)

    setup = DeviceSetupDialog(settings=settings)
    if setup.exec() != DeviceSetupDialog.DialogCode.Accepted:
        sys.exit(0)

    config = setup.get_config()
    window = SidebarPanel(config, settings)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()