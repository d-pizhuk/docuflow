import sys
from PySide6.QtWidgets import QApplication
from session.device_setup_dialog import DeviceSetupDialog
from session.sidebar_panel import SidebarPanel


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("DocuFlow")
    app.setStyle("Fusion")

    setup = DeviceSetupDialog()
    if setup.exec() != DeviceSetupDialog.DialogCode.Accepted:
        sys.exit(0)

    config = setup.get_config()

    window = SidebarPanel(config)
    window.session_finished.connect(app.quit)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
