#!/usr/bin/env python3
"""
main.py — Face Recognition System CLI
======================================
Interactive command-line interface for all face recognition operations.

Usage:
    python main.py                  # Interactive menu
    python main.py --live           # Start live recognition
    python main.py --register       # Register new person
    python main.py --web            # Launch web dashboard
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))


def clear():
    os.system('cls' if os.name == 'nt' else 'clear')


def banner():
    print("""
╔═══════════════════════════════════════════════════════╗
║          FACE RECOGNITION SYSTEM  v1.0                ║
║          Deep Learning · CNN · Real-Time               ║
╚═══════════════════════════════════════════════════════╝
""")


def menu():
    print("""
  [1]  Live webcam recognition
  [2]  Recognize faces in an image file
  [3]  Register person (from image file)
  [4]  Register person (from webcam)
  [5]  List registered persons
  [6]  Delete a person
  [7]  Train / fine-tune model
  [8]  Evaluate accuracy
  [9]  Launch web dashboard
  [0]  Exit
""")


def main():
    parser = argparse.ArgumentParser(description='Face Recognition System')
    parser.add_argument('--live', action='store_true', help='Start live recognition immediately')
    parser.add_argument('--register', action='store_true', help='Start registration flow')
    parser.add_argument('--web', action='store_true', help='Launch web dashboard')
    parser.add_argument('--image', type=str, help='Recognize faces in a specific image')
    parser.add_argument('--threshold', type=float, default=0.6, help='Recognition threshold (0-1)')
    parser.add_argument('--detector', choices=['haar', 'dnn'], default='haar')
    args = parser.parse_args()

    # Fast path flags
    if args.live:
        from face_engine import FaceRecognitionSystem
        frs = FaceRecognitionSystem(detector_method=args.detector, recognition_threshold=args.threshold)
        frs.run_live_recognition()
        return

    if args.web:
        os.system(f"{sys.executable} web_app.py")
        return

    if args.image:
        import cv2
        from face_engine import FaceRecognitionSystem
        frs = FaceRecognitionSystem(detector_method=args.detector, recognition_threshold=args.threshold)
        annotated, results = frs.recognize_image(args.image)
        if annotated is not None:
            for r in results:
                print(f"  → {r['name']} ({r['confidence']}%) at bbox {r['bbox']}")
            cv2.imshow("Recognition Result", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    # Interactive menu
    clear()
    banner()

    # Lazy-load system
    frs = None

    def get_frs():
        nonlocal frs
        if frs is None:
            print("\n  Initializing system...", end='', flush=True)
            from face_engine import FaceRecognitionSystem
            frs = FaceRecognitionSystem(detector_method=args.detector,
                                        recognition_threshold=args.threshold)
            print(" done.\n")
        return frs

    while True:
        menu()
        choice = input("  Select option: ").strip()

        if choice == '1':
            get_frs().run_live_recognition()

        elif choice == '2':
            import cv2
            path = input("  Image path: ").strip()
            if not os.path.exists(path):
                print("  File not found.")
                continue
            annotated, results = get_frs().recognize_image(path)
            if results:
                print(f"\n  Found {len(results)} face(s):")
                for r in results:
                    print(f"    → {r['name']} — {r['confidence']}%")
                cv2.imshow("Recognition Result", annotated)
                print("  Press any key to close...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("  No faces detected or file error.")

        elif choice == '3':
            path = input("  Image path: ").strip()
            name = input("  Person name: ").strip()
            ok, msg = get_frs().register_from_image(path, name)
            print(f"  {'✓' if ok else '✗'} {msg}")

        elif choice == '4':
            name = input("  Person name: ").strip()
            num = int(input("  Number of samples to capture (default 10): ").strip() or 10)
            ok, msg = get_frs().register_from_webcam(name, num_samples=num)
            print(f"  {'✓' if ok else '✗'} {msg}")

        elif choice == '5':
            get_frs().list_registered()
            # pause is handled by the "Press Enter" at loop bottom

        elif choice == '6':
            name = input("  Name to delete: ").strip()
            ok = get_frs().delete_person(name)
            print(f"  {'✓ Deleted' if ok else '✗ Not found'}: {name}")

        elif choice == '7':
            print("\n  Starting training pipeline...")
            from train_model import train
            data_dir = input("  Data directory (default data/raw): ").strip() or "data/raw"
            epochs = int(input("  Epochs (default 30): ").strip() or 30)
            train(data_dir=data_dir, epochs=epochs)

        elif choice == '8':
            from evaluate import evaluate_on_dataset, evaluate_from_database
            data_dir = Path("data/raw")
            if data_dir.exists() and any(data_dir.iterdir()):
                evaluate_on_dataset(get_frs(), test_dir=str(data_dir))
            else:
                print("\n  data/raw not found — running self-test on registered database instead.")
                evaluate_from_database(get_frs())

        elif choice == '9':
            print("\n  Launching web dashboard at http://localhost:5000 ...")
            os.system(f"{sys.executable} web_app.py")

        elif choice == '0':
            print("\n  Goodbye!\n")
            break

        else:
            print("  Invalid option.")

        input("\n  Press Enter to continue...")
        clear()
        banner()


if __name__ == '__main__':
    main()