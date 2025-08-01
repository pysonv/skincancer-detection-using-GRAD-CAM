import argparse
from skin_cancer_detection.src.model import SkinCancerDetector
from skin_cancer_detection.src.train import train_model
from skin_cancer_detection.src.visualize import visualize_grad_cam

def main():
    parser = argparse.ArgumentParser(description='Skin Cancer Detection with Grad-CAM')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'predict'],
                        help='Mode to run: train or predict')
    parser.add_argument('--image_path', type=str, help='Path to the image for prediction')
    parser.add_argument('--train_dir', type=str, default='data/train', help='Path to training data')
    parser.add_argument('--val_dir', type=str, default='data/validation', help='Path to validation data')
    parser.add_argument('--model_path', type=str, default='best_model.h5', help='Path to saved model file')
    parser.add_argument('--epochs', type=int, default=25, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')

    args = parser.parse_args()

    detector = SkinCancerDetector()

    if args.mode == 'train':
        print("Mode: Training")
        detector.build_model()
        train_model(detector, args.train_dir, args.val_dir, args.epochs, args.batch_size, args.model_path)
        print(f"Training complete. Model saved to {args.model_path}")

    elif args.mode == 'predict':
        if not args.image_path:
            raise ValueError("An image path must be provided for prediction.")
        print(f"Mode: Predicting on {args.image_path}")
        detector.model.load_weights(args.model_path)
        visualize_grad_cam(detector, args.image_path)

if __name__ == '__main__':
    main()
