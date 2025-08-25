import torch
import torchvision.transforms as transforms
import torchvision.models as models
import cv2

class CourtLineDetector:
    def __init__(self, model_path):
        # ? Clear
        # Load the YOLO model
        # self.model = models.resnet50(pretrained=True)
        self.model = models.efficientnet_b0(pretrained=True)
        
        
        # self.model.fc = torch.nn.Linear(self.model.fc.in_features, 28)
        num_features = self.model.classifier[1].in_features
        self.model.classifier[1] = torch.nn.Linear(num_features, 28)

        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

        self.model.eval()

        # ? Clear         
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            # Augmentation
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def show_model_structure(self):
        # --- For ResNet50 (for comparison) ---
        model_resnet = models.resnet50(pretrained=True)
        print("--- ResNet50 Structure ---")
        print(model_resnet)
        # Scroll to the very bottom, you'll see something like:
        # (fc): Linear(in_features=2048, out_features=1000, bias=True)
        # The name in parentheses (fc) is how you access it.

        # --- For EfficientNet-B0 ---
        model_efficientnet = models.efficientnet_b0(pretrained=True)
        print("\n--- EfficientNet-B0 Structure ---")
        print(model_efficientnet)
        # Scroll to the very bottom, you'll see something like:
        # (classifier): Sequential(
        #   (0): BatchNorm1d(1280, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #   (1): Linear(in_features=1280, out_features=1000, bias=True)
        # )
        # Here, 'classifier' is a 'Sequential' module, and the actual Linear layer
        # is the second element (index 1) within that Sequential module.
        # So you access it as model.classifier[1].

        # --- For DenseNet121 ---
        model_densenet = models.densenet121(pretrained=True)
        print("\n--- DenseNet121 Structure ---")
        print(model_densenet)
        # Scroll to the very bottom, you'll see something like:
        # (classifier): Linear(in_features=1024, out_features=1000, bias=True)
        # Here, 'classifier' is directly the Linear layer. So you access it as model.classifier.

    def predict(self, image):
        """
        Predict court lines in a single frame.

        Args: 
            frame (numpy.ndarray): The video frame to process.

        Returns:
            list: List of predicted court line classes.
        """
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(img_rgb).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(img_tensor)

        keypoints = outputs.squeeze().cpu().numpy()
        original_height, original_width = img_rgb.shape[:2]
        keypoints[::2] *= original_width/224.0
        keypoints[1::2] *= original_height/224.0

        return keypoints

    def draw_keypoints(self, image, keypoints):
        for i in range(0, len(keypoints), 2):
            x = int(keypoints[i])
            y = int(keypoints[i + 1])
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(image, str(i//2), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        return image
    
    def draw_keypoints_on_video(self, video_frames, keypoints):
        """
        Draw keypoints on video frames.

        Args:
            video_frames (list): List of video frames.
            keypoints_list (list): keypoints for video.
            
        Returns:
            list: List of video frames with keypoints drawn.
        """
        output_video_frames = []
        for frame in video_frames:
            frame = self.draw_keypoints(frame, keypoints)
            output_video_frames.append(frame)
        return output_video_frames
    
    # draw shapes on court
    def draw_shapes_on_court(self, image, keypoints):
        # Example: Draw lines between keypoints
        for i in range(0, len(keypoints) - 2, 2):
            x1, y1 = int(keypoints[i]), int(keypoints[i + 1])
            x2, y2 = int(keypoints[i + 2]), int(keypoints[i + 3])
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        return image

    
    
