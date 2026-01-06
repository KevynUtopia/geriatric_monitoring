import sys
import os

# Add project root directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """Main function"""
    try:
        from PyQt5.QtWidgets import QApplication
        from app.main_window import DetectionApp
        
        # Create QApplication
        app = QApplication(sys.argv)
        app.setApplicationName("AI Detection Template")
        app.setApplicationVersion("1.0.0")
        
        # Create and show main window
        window = DetectionApp()
        window.show()
        
        # Run application
        sys.exit(app.exec_())
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please ensure all dependencies are correctly installed")
        print("Run: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"Startup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

