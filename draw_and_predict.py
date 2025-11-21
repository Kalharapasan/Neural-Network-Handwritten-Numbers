#!/usr/bin/env python3
"""
Interactive Drawing Interface for Handwritten Digit Recognition
Allows users to draw digits on screen and get real-time predictions
"""

import tkinter as tk
from tkinter import ttk, messagebox
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageTk
import io

class DigitRecognizer(nn.Module):
    """Neural Network Architecture for digit recognition"""
    def __init__(self):
        super(DigitRecognizer, self).__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)

class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Digit Recognition - Draw & Predict")
        self.root.geometry("800x600")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()
        
        # Drawing variables
        self.canvas_size = 280
        self.brush_size = 12
        self.drawing = False
        self.last_x = None
        self.last_y = None
        
        # Create PIL image for drawing
        self.pil_image = Image.new("RGB", (self.canvas_size, self.canvas_size), "white")
        self.pil_draw = ImageDraw.Draw(self.pil_image)
        
        self.setup_ui()
        
    def load_model(self):
        """Load the pre-trained model"""
        try:
            model = DigitRecognizer()
            model.load_state_dict(torch.load('digit_recognizer.pth', 
                                           map_location=self.device, 
                                           weights_only=True))
            model.to(self.device)
            model.eval()
            print(f"Model loaded successfully on {self.device}")
            return model
        except FileNotFoundError:
            messagebox.showerror("Error", 
                               "Model file 'digit_recognizer.pth' not found!\n"
                               "Please train the model first using the Jupyter notebook.")
            return None
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            return None
    
    def setup_ui(self):
        """Setup the user interface"""
        # Title
        title_frame = tk.Frame(self.root, bg='#f0f0f0')
        title_frame.pack(pady=10)
        
        title_label = tk.Label(title_frame, 
                              text="🖊️ Draw a Digit (0-9) and See AI Prediction", 
                              font=("Arial", 16, "bold"),
                              bg='#f0f0f0',
                              fg='#333333')
        title_label.pack()
        
        # Main content frame
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(expand=True, fill='both', padx=20, pady=10)
        
        # Left side - Drawing canvas
        canvas_frame = tk.LabelFrame(main_frame, 
                                    text="Drawing Canvas", 
                                    font=("Arial", 12, "bold"),
                                    bg='#f0f0f0',
                                    fg='#333333')
        canvas_frame.pack(side='left', padx=(0, 10), pady=5, fill='both', expand=True)
        
        # Canvas
        self.canvas = tk.Canvas(canvas_frame, 
                               width=self.canvas_size, 
                               height=self.canvas_size,
                               bg='white', 
                               cursor='pencil')
        self.canvas.pack(padx=10, pady=10)
        
        # Bind drawing events
        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)
        
        # Canvas controls
        canvas_controls = tk.Frame(canvas_frame, bg='#f0f0f0')
        canvas_controls.pack(pady=5)
        
        clear_btn = tk.Button(canvas_controls, 
                             text="🗑️ Clear Canvas",
                             command=self.clear_canvas,
                             font=("Arial", 10, "bold"),
                             bg='#ff4444',
                             fg='white',
                             cursor='hand2',
                             relief='raised',
                             borderwidth=2)
        clear_btn.pack(side='left', padx=5)
        
        predict_btn = tk.Button(canvas_controls, 
                               text="🔍 Predict Now",
                               command=self.predict_digit,
                               font=("Arial", 10, "bold"),
                               bg='#4CAF50',
                               fg='white',
                               cursor='hand2',
                               relief='raised',
                               borderwidth=2)
        predict_btn.pack(side='left', padx=5)
        
        # Right side - Prediction results
        results_frame = tk.LabelFrame(main_frame, 
                                     text="Prediction Results", 
                                     font=("Arial", 12, "bold"),
                                     bg='#f0f0f0',
                                     fg='#333333')
        results_frame.pack(side='right', padx=(10, 0), pady=5, fill='both')
        
        # Prediction display
        self.prediction_label = tk.Label(results_frame, 
                                        text="Draw a digit to see prediction",
                                        font=("Arial", 24, "bold"),
                                        bg='#f0f0f0',
                                        fg='#666666')
        self.prediction_label.pack(pady=20)
        
        # Confidence display
        self.confidence_label = tk.Label(results_frame, 
                                        text="Confidence: --",
                                        font=("Arial", 14),
                                        bg='#f0f0f0',
                                        fg='#666666')
        self.confidence_label.pack(pady=5)
        
        # All predictions frame
        predictions_frame = tk.LabelFrame(results_frame, 
                                         text="All Predictions", 
                                         font=("Arial", 10, "bold"),
                                         bg='#f0f0f0')
        predictions_frame.pack(pady=10, padx=10, fill='both', expand=True)
        
        # Create labels for all digits (0-9)
        self.digit_labels = {}
        for i in range(10):
            frame = tk.Frame(predictions_frame, bg='#f0f0f0')
            frame.pack(fill='x', padx=5, pady=2)
            
            digit_label = tk.Label(frame, 
                                  text=f"Digit {i}:",
                                  font=("Arial", 10),
                                  bg='#f0f0f0',
                                  width=8,
                                  anchor='w')
            digit_label.pack(side='left')
            
            prob_label = tk.Label(frame, 
                                 text="0.0%",
                                 font=("Arial", 10),
                                 bg='#f0f0f0',
                                 anchor='e')
            prob_label.pack(side='right')
            
            self.digit_labels[i] = prob_label
        
        # Instructions
        instructions_frame = tk.Frame(self.root, bg='#f0f0f0')
        instructions_frame.pack(pady=5)
        
        instructions = tk.Label(instructions_frame,
                               text="💡 Tips: Draw clearly in the center • Use thick strokes • Try different digits",
                               font=("Arial", 10),
                               bg='#f0f0f0',
                               fg='#666666')
        instructions.pack()
        
        # Status bar
        self.status_label = tk.Label(self.root, 
                                    text=f"Model loaded on {self.device}" if self.model else "Model not loaded",
                                    font=("Arial", 9),
                                    bg='#e0e0e0',
                                    fg='#333333',
                                    anchor='w',
                                    relief='sunken')
        self.status_label.pack(side='bottom', fill='x')
    
    def start_drawing(self, event):
        """Start drawing on canvas"""
        self.drawing = True
        self.last_x = event.x
        self.last_y = event.y
    
    def draw(self, event):
        """Draw on canvas"""
        if self.drawing and self.last_x and self.last_y:
            # Draw on tkinter canvas
            self.canvas.create_line(self.last_x, self.last_y, 
                                   event.x, event.y,
                                   width=self.brush_size, 
                                   fill='black', 
                                   capstyle=tk.ROUND, 
                                   smooth=tk.TRUE)
            
            # Draw on PIL image
            self.pil_draw.line([self.last_x, self.last_y, event.x, event.y], 
                              fill='black', 
                              width=self.brush_size)
            
            self.last_x = event.x
            self.last_y = event.y
            
            # Real-time prediction (optional, can be resource intensive)
            # Uncomment the next line for real-time prediction while drawing
            # self.predict_digit(show_status=False)
    
    def stop_drawing(self, event):
        """Stop drawing and make prediction"""
        self.drawing = False
        self.last_x = None
        self.last_y = None
        
        # Make prediction after drawing stops
        if self.model:
            self.predict_digit()
    
    def clear_canvas(self):
        """Clear the drawing canvas"""
        self.canvas.delete("all")
        self.pil_image = Image.new("RGB", (self.canvas_size, self.canvas_size), "white")
        self.pil_draw = ImageDraw.Draw(self.pil_image)
        
        # Reset prediction display
        self.prediction_label.config(text="Draw a digit to see prediction", fg='#666666')
        self.confidence_label.config(text="Confidence: --", fg='#666666')
        
        # Reset all digit probabilities
        for i in range(10):
            self.digit_labels[i].config(text="0.0%", fg='#666666')
        
        self.status_label.config(text="Canvas cleared - ready for new digit")
    
    def preprocess_image(self):
        """Preprocess the canvas image for model input"""
        # Convert to grayscale
        img = self.pil_image.convert('L')
        
        # Apply slight blur to smooth the drawing
        img = img.filter(ImageFilter.GaussianBlur(radius=1))
        
        # Resize to 28x28
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize
        img_array = np.array(img)
        
        # Invert colors (white background to black, black drawing to white)
        img_array = 255 - img_array
        
        # Normalize to [0, 1]
        img_array = img_array.astype(np.float32) / 255.0
        
        # Apply MNIST normalization
        img_array = (img_array - 0.1307) / 0.3081
        
        # Convert to PyTorch tensor
        tensor = torch.FloatTensor(img_array).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        
        return tensor
    
    def predict_digit(self, show_status=True):
        """Make prediction on the drawn digit"""
        if not self.model:
            return
        
        try:
            # Preprocess the image
            input_tensor = self.preprocess_image().to(self.device)
            
            # Make prediction
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                predicted_digit = output.argmax(dim=1).item()
                confidence = probabilities[0][predicted_digit].item() * 100
            
            # Update prediction display
            color = '#2E8B57' if confidence > 70 else '#FF6347' if confidence > 40 else '#FF4500'
            self.prediction_label.config(text=f"Predicted: {predicted_digit}", fg=color)
            self.confidence_label.config(text=f"Confidence: {confidence:.1f}%", fg=color)
            
            # Update all digit probabilities
            probs = probabilities[0].cpu().numpy()
            for i in range(10):
                prob_percent = probs[i] * 100
                color = '#2E8B57' if i == predicted_digit else '#666666'
                self.digit_labels[i].config(text=f"{prob_percent:.1f}%", fg=color)
            
            if show_status:
                self.status_label.config(text=f"Prediction: {predicted_digit} (Confidence: {confidence:.1f}%)")
        
        except Exception as e:
            messagebox.showerror("Prediction Error", f"Error making prediction: {str(e)}")

def main():
    """Main function to run the application"""
    root = tk.Tk()
    
    # Set window icon (optional)
    try:
        # You can add an icon file here if available
        # root.iconbitmap('icon.ico')
        pass
    except:
        pass
    
    app = DrawingApp(root)
    
    # Center window on screen
    root.update_idletasks()
    x = (root.winfo_screenwidth() - root.winfo_width()) // 2
    y = (root.winfo_screenheight() - root.winfo_height()) // 2
    root.geometry(f"+{x}+{y}")
    
    # Start the application
    root.mainloop()

if __name__ == "__main__":
    main()