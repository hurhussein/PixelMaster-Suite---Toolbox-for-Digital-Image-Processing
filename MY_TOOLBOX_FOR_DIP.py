import tkinter as tk
from tkinter import filedialog ,messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import easyocr
import imutils
import matplotlib.pyplot as plt 

# creat gui class
class ImageProcessingToolbox:
    # self to start out
    # master refer to window i will creat
    def __init__(self, master):
        self.master = master
        self.master.title("Image Processing Toolbox")
        
        # Set the size and position of the window(W*H)
        self.master.geometry("1600x750")
        #self.master.resizable(False, False)  # Fixed window size
        # /////////////////colors ///////////////////////////////////////////////
        
        self.b_color='#a85163'           #button color
        self.f_color='#142f30'           # frame , font and window color
        self.master.config(bg=self.f_color)  # Set window background
      
        

        # Variables for image processing
        self.image = None
        self.display_image_original = None
        self.display_image_processed = None
        self.zoom_factor = 1.0

        # Frames to split the window
        self.frame_B = tk.Frame(self.master, width=200, height=690, bg=self.f_color)
        self.frame_B.place(x=0, y=5)

        self.frame_B1 = tk.Frame(self.master, width=200, height=690, bg=self.f_color)
        self.frame_B1.place(x=220, y=50)
        
        
        self.frame_Bf = tk.Frame(self.master, width=200, height=690, bg=self.f_color)
        self.frame_Bf.place(x=410, y=5)
        self.create_label(self.frame_Bf, " Filters ")
        
        self.frame_B_od = tk.Frame(self.master, width=200, height=400, bg=self.f_color)
        self.frame_B_od.place(x=410, y=320)
        self.create_label(self.frame_B_od, " Object detection ")
        
        
        self.frame_original = tk.Frame(self.master, width=400, height=690, bg=self.f_color)
        self.frame_original.place(x=730, y=10)
        self.create_label(self.frame_original, " orginal image")
        
        self.frame_processing = tk.Frame(self.master, width=390, height=690, bg=self.f_color)
        self.frame_processing.place(x=1130, y=10)
        self.create_label(self.frame_processing, " image after processing")

        


        # Canvas widgets
        self.canvas_original = tk.Canvas(self.frame_original, width=400, height=670, bg=self.f_color)
        self.canvas_original.pack()

        self.canvas_processed = tk.Canvas(self.frame_processing, width=390, height=670 ,bg=self.f_color)
        self.canvas_processed.pack()
# ///////////////////////     Buttons  ///////////////////////////////////////////////////////////////////////////
      # Buttons for image processing functions
        self.create_button(self.frame_B, "Open Image", self.open_image)
        self.create_button(self.frame_B, "Save", self.save_processed_image)
        # /////////////////////point pro ////////////////////////////////////////////
        self.create_button(self.frame_B, "Zoom In", self.zoom_in)
        self.create_button(self.frame_B, "Zoom Out", self.zoom_out)

        self.create_button(self.frame_B,"translation",self.translation)

        self.create_button(self.frame_B, "Rotate", self.rotate_image)
        self.create_button(self.frame_B, "Skewing", self.skewing)
        self.create_button(self.frame_B, "Scaling", self.scaling)
        self.create_button(self.frame_B, "fliping", self.reflecting)
       
        self.create_button(self.frame_B, "Adjust Brightness/Contrast", self.gamma_correction)
        self.create_button(self.frame_B, "Improve image contrast", self.histogram_equalization)
        self.create_button(self.frame_B, "Invert Colors", self.negative_transformation)
        self.create_button(self.frame_B, "Log Transformations", self.logarithmic_transformation)
        
        self.create_button(self.frame_B, "Bit Plane Slicing", self.bit_plane_slicing)
        self.create_button(self.frame_B, "Gray_level_slicing", self.gray_level_slicing)
        # /////////////////////////// filter Buttons ////////////////
        self.create_button(self.frame_Bf, "smoothing_filter", self.apply_smoothing_filter)
        self.create_button(self.frame_Bf, "Blur_filter", self.apply_median_blur)
        # self.create_button(self.frame_Bf, "Sharpening_filter", self.apply_sharpening_filter)
        self.create_button(self.frame_Bf, "Horizontal edge detection", self.apply_h_edges)
        self.create_button(self.frame_Bf, " vertical edge detectio", self.apply_v_edges)
        # self.create_button(self.frame_Bf, "Remove spark noise and get less blurring", self.apply_frequency_domain_enhancement)
        self.create_button(self.frame_Bf, " separating objects from the background", self.apply_thresholding)
        self.create_button(self.frame_Bf, "DFT", self.apply_dft)
        # /////////////////////////////////////////// button for more options( detection) /////////////////////////////////////////////
        self.create_button(self.frame_B_od, " Object Detection in images ", self.object_detection_image)
        self.create_button(self.frame_B_od, " Object Detection in video  ", self.object_detection_video)
        self.create_button(self.frame_B_od ," Car-plate-recognition", self.apply_ocr)



        
        # ///////////////////////Entry widgets ///////////////////////
        self.tx_entry = self.create_entry(self.frame_B1, "Translate X:", 'tx_entry', default_value='0')
        
      
        self.ty_entry = self.create_entry(self.frame_B1, "Translate Y:", 'ty_entry', default_value='0')

        self.rotation_entry = self.create_entry(self.frame_B1, "Rotation Angle:", 'rotation_entry', default_value='45')
        self.skew_entry = self.create_entry(self.frame_B1, "Skew:", 'skew_entry', default_value='0')
        self.scale_entry = self.create_entry(self.frame_B1, "Scale:", 'scale_entry')
        self.flip_entry=self.create_entry(self.frame_B1,"fliping",'flip_entry',default_value='0')

        self.gamma_entry=self.create_entry(self.frame_B1, "Brightness/Contrast :", 'gamma_entry',default_value='0')
   
       
    
    #    self.create_entry(self.frame_buttons, "Lower Gray Level:", 'lower_gray_level_entry', default_value='0')
    #    self.create_entry(self.frame_buttons, "Upper Gray Level:", 'upper_gray_level_entry', default_value='255')
    
#///////////////////////////// gui func////////////////

#  ////////////////////////////////////button func//////////////////////

    def create_button(self, frame, button_text, command):
        # Create buttons with specified text and command
        button = tk.Button(frame, text=button_text, command=command, pady=1, font=('Arial', 12))
        
        # Styling enhancements with cozy and feminine colors
        button.config(bg= self.b_color , fg=self.f_color)  
        button.pack(fill=tk.X, padx=10, pady=5, ipady=2)  


        return button
        
        
    def create_entry(self, frame, label_text,entry_n, default_value='0'):
    # Create a label with the specified text
       label = tk.Label(frame, text=label_text, pady=5, font=('Arial', "14","bold" ),fg=self.b_color,bg=self.f_color)
       label.pack(anchor='w')  # Pack the label to the left side

    # Create an entry widget with a default width and font
       entry = tk.Entry(frame, width=5, font=('Arial', 12))
    
    # Insert the default value into the entry widget
       entry.insert(tk.END, default_value)
    
       entry.pack()  # Pack the entry widget

    # Return the created entry widget
       return entry
   
    def create_label(self, frame, text):
        label = tk.Label(frame, text=text, font=("Arial", "15","bold"),bg=self.f_color, fg=self.b_color)
        label.pack(pady=1)


    #   //////////////////////display fu///////////////////////////////////////

    def open_image(self):
        # Open a file dialog to select an image file
        file_path = filedialog.askopenfilename()

        # Check if a file path was selected (user didn't cancel the file dialog)
        if file_path:
            # Read the image from the selected file path using OpenCV
            
            self.r_image = cv2.imread(file_path)
            
            self.image = cv2.resize(self.r_image, (400, 400)) 
            
            # Convert the OpenCV image to a Tkinter PhotoImage
            self.display_image_original = self.convert_cv_image_to_tk_image( self.image )
            
            # Display the converted image on the original canvas
            self.display_image_on_canvas(self.display_image_original, self.canvas_original)
            
            # ////////////////creat label///////////////////////////////

            
            
            
            
            
# ////////////////////////// display pic /////////////////////////////////////////////////
    def display_image_on_canvas(self, image, canvas):
        # Clear the canvas to remove any existing content
        canvas.delete("all")
        
        # Create a new image item on the canvas, anchored at the top-left corner
        canvas.create_image(0, 0, anchor=tk.NW, image=image)

    def convert_cv_image_to_tk_image(self, cv_image):
        # Convert the BGR image to gray
        image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # Create a PIL Image from the RGB image
        image_pil = Image.fromarray(image_rgb)
        
        # Convert the PIL Image to a Tkinter PhotoImage
        return ImageTk.PhotoImage(image_pil)
    # /////////////////// save //////////////////////////////////////////////////////////////////////////
    def save_processed_image(self):
        if self.display_image_processed is not None:
            # Open a file dialog to get the file path for saving the image
            file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                    filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
            if file_path:
                # Get the Tkinter PhotoImage as a PIL Image
                pil_image = ImageTk.getimage(self.display_image_processed)

                # Save the PIL Image to the specified file path
                pil_image.save(file_path)

                messagebox.showinfo("Saved", f"Image saved successfully at {file_path}.")
        else:
            messagebox.showinfo("Error", "No processed image to save")


# ///////////////////// DIP FUN////////////////////////////////////////////////////////////////////////////////////////


# //////////////zooom//////////////
    def zoom_image(self, factor):
        # Check if there is an image loaded
        if self.image is not None:
            # Get the dimensions of the original image
            rows, cols, _ = self.image.shape

            # Resize the image based on the zoom factor
            resized_image = cv2.resize(self.image, (int(cols * factor), int(rows * factor)))

            # Convert the resized image to Tkinter PhotoImage
            self.display_image_processed = self.convert_cv_image_to_tk_image(resized_image)

            # Display the processed image on the canvas
            self.display_image_on_canvas(self.display_image_processed, self.canvas_processed)

            # Update the zoom factor
            self.zoom_factor = factor
            

    def zoom_in(self):
        if self.image is not None:
            # Zoom in the image by increasing the zoom factor
            self.zoom_factor *= 1.2
            
            # Call the zoom_image function with the updated zoom factor
            self.zoom_image(self.zoom_factor)
            # cv2.imshow("self.display_image_processed")
            print("test")
        else:
                         # Display an error message if no image is loaded
            messagebox.showinfo("Error", "No image loaded")
            

    def zoom_out(self):
        if self.image is not None:
            # Zoom out the image by decreasing the zoom factor
            self.zoom_factor /= 1.2
            
            # Call the zoom_image function with the updated zoom factor
            self.zoom_image(self.zoom_factor)
        else:
                         # Display an error message if no image is loaded
            messagebox.showinfo("Error", "No image loaded")
        

# ///////////////////// rotate_image(//////////////////////////
    def rotate_image(self):
        try:
            if self.image is not None:
                # Attempt to convert user input to float for the rotation angle
                angle = float(self.rotation_entry.get())

                # Get image dimensions
                rows, cols, _ = self.image.shape

                # Create a rotation matrix based on the user-specified angle
                M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)

                # Apply the rotation to the image
                rotated_image = cv2.warpAffine(self.image, M, (cols, rows))

                # Convert the rotated image to Tkinter PhotoImage
                self.display_image_processed = self.convert_cv_image_to_tk_image(rotated_image)

                # Display the rotated image on the processed image canvas
                self.display_image_on_canvas(self.display_image_processed, self.canvas_processed)
            else:
                             # Display an error message if no image is loaded
                 messagebox.showinfo("Error", "No image loaded")

        except ValueError:
            # If a ValueError occurs (e.g., non-numeric input), show an error message
            messagebox.showinfo("Error", "Please enter a valid numeric value for the rotation angle.")
            
            
            
# ////////////////////////////////translation/////////////////////////////////////////
    def translation(self):
        
        try:
            
             if self.image is not None:
                 
                # Attempt to convert user input to integers
                tx = int(self.tx_entry.get())
                ty = int(self.ty_entry.get())

                # Get image dimensions
                rows, cols, _ = self.image.shape

                # Create a translation matrix
                M = np.float32([[1, 0, tx], [0, 1, ty]])

                # Apply the translation to the image
                translated_image = cv2.warpAffine(self.image, M, (cols, rows))

                # Convert the translated image to Tkinter PhotoImage
                self.display_image_processed = self.convert_cv_image_to_tk_image(translated_image)

                # Display the translated image on the processed image canvas
                self.display_image_on_canvas(self.display_image_processed, self.canvas_processed)
                
             else:  
                             # Display an error message if no image is loaded
                messagebox.showinfo("Error", "No image loaded")

        except ValueError:
        # If a ValueError occurs (e.g., non-numeric input), show an error message
         
             messagebox.showinfo("Error", "Please enter a valid numeric value for the tx and ty.")


            
            
            
            
            
            
    #  ///////////////////////////////////////////skewing///////////////////////////////////////////       
    def skewing(self):
        try:
            if self.image is not None:
            # Attempt to convert user input to float for the skew value
                skew_value = float(self.skew_entry.get())

            # Get image dimensions
                rows, cols, _ = self.image.shape

            # Create a skew matrix
                M = np.float32([[1, skew_value, 0], [0, 1, 0]])

            # Apply the skew to the image
                skewed_image = cv2.warpAffine(self.image, M, (cols, rows))

            # Convert the skewed image to Tkinter PhotoImage
                self.display_image_processed = self.convert_cv_image_to_tk_image(skewed_image)

            # Display the skewed image on the processed image canvas
                self.display_image_on_canvas(self.display_image_processed, self.canvas_processed)
            else:    
                             # Display an error message if no image is loaded
                messagebox.showinfo("Error", "No image loaded")
            
          
        except ValueError:
             messagebox.showinfo("Error", "Please enter a valid numeric value forskewing value .")
             
             
             
             
             
             
             
             
             
            #  /////////////////////////////////////scaling/////////////////////////////////////////
    
    def scaling(self):
        try:
            if self.image is not None:
                scale_value = float(self.scale_entry.get())
                rows, cols, _ = self.image.shape
                scaled_image = cv2.resize(self.image, (int(cols * scale_value), int(rows * scale_value)))
                self.display_image_processed = self.convert_cv_image_to_tk_image(scaled_image)
                self.display_image_on_canvas(self.display_image_processed, self.canvas_processed)
            else:
                             # Display an error message if no image is loaded
                messagebox.showinfo("Error", "No image loaded")
        except ValueError:
            messagebox.showinfo("Error", "Please enter a valid numeric value for the scaling value.")

        # ///////////////////// fliping/////////////////////////////////////////////////////////////// 
    def reflecting(self):
         try:
            if self.image is not None:
                
                flip_value=int(self.flip_entry.get())
                reflected_image = cv2.flip(self.image,flip_value )
                self.display_image_processed = self.convert_cv_image_to_tk_image(reflected_image)
                self.display_image_on_canvas(self.display_image_processed, self.canvas_processed)
            else:
                             # Display an error message if no image is loaded
                messagebox.showinfo("Error", "No image loaded")
         except ValueError:
            messagebox.showinfo("Error", "Please enter a valid numeric value for the scaling value.")
            # /////////////////////////////////histogram///////////////////////////////
    def histogram_equalization(self):
        if self.image is not None:
            
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            equalized_image = cv2.equalizeHist(gray_image)
            equalized_image_bgr = cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2BGR)
            self.display_image_processed = self.convert_cv_image_to_tk_image(equalized_image_bgr)
            self.display_image_on_canvas(self.display_image_processed, self.canvas_processed)
        else:
                         # Display an error message if no image is loaded
            messagebox.showinfo("Error", "No image loaded")
            # ////////////////////////////////// gamma//////////////////////////////////////////
    def gamma_correction(self):
        try:
            gamma_value = float(self.gamma_entry.get())
            
           
            # Check if there is an image loaded
            if self.image is not None:
                # Convert the BGR image to grayscale
                gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

                # Perform gamma correction on the grayscale image
                gamma_corrected_image = np.clip((gray_image / 255.0) ** gamma_value * 255.0, 0, 255).astype(np.uint8)

                # Convert the gamma-corrected image to Tkinter PhotoImage
                self.display_image_processed = self.convert_cv_image_to_tk_image(gamma_corrected_image)

                # Display the processed image on the canvas
                self.display_image_on_canvas(self.display_image_processed, self.canvas_processed)
            else:
                             # Display an error message if no image is loaded
                messagebox.showinfo("Error", "No image loaded")

        except ValueError:
            messagebox.showinfo("Error", "Please enter a valid numeric value for gamma.")
# //////////////////////////////////////negative_transformation//////////////////////

    def negative_transformation(self):
            # Check if there is an image loaded
            if self.image is not None:
                # Convert the BGR image to grayscale
                gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

                # Calculate the maximum intensity level (L)
                L = 256

                # Perform the image negative transformation
                negative_image = L - 1 - gray_image

                # Convert the negative image to Tkinter PhotoImage
                self.display_image_processed = self.convert_cv_image_to_tk_image(negative_image)

                # Display the processed image on the canvas
                self.display_image_on_canvas(self.display_image_processed, self.canvas_processed)
            else:
                             # Display an error message if no image is loaded
                messagebox.showinfo("Error", "No image loaded")


            # ////////////////////////////logarithmic_transformation  (natural log=ln x) ////////////////
    def logarithmic_transformation(self):
        # Check if there is an image loaded
        if self.image is not None:
            # Convert the BGR image to grayscale
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

            # Define the constant for scaling (you can adjust this)
            c = 1.0

            # Perform the logarithmic transformation
            logarithmic_image = c * np.log1p(gray_image)

            # Normalize the image to the range [0, 255]
            logarithmic_image = ((logarithmic_image - logarithmic_image.min()) /
                                 (logarithmic_image.max() - logarithmic_image.min()) * 255).astype(np.uint8)

            # Convert the logarithmic image to Tkinter PhotoImage
            self.display_image_processed = self.convert_cv_image_to_tk_image(logarithmic_image)

            # Display the processed image on the canvas
            self.display_image_on_canvas(self.display_image_processed, self.canvas_processed)
        else:
                         # Display an error message if no image is loaded
            messagebox.showinfo("Error", "No image loaded")
            
            
            
# ////////////////////////////Bit Plane Slicing/////////////////////////////////////////////////
    def bit_plane_slicing(self):
        
            # Check if there is an image loaded
            if self.image is not None:
                # Convert the BGR image to grayscale
                gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

                # Iterate over all bit planes
                for bit_position in range(8):
                    # Extract the specified bit plane
                    bit_plane_image = (gray_image >> bit_position) & 1

                    # Multiply by 255 to display the result
                    bit_plane_image *= 255

                    # Convert the bit plane image to Tkinter PhotoImage
                    self.display_image_processed = self.convert_cv_image_to_tk_image(bit_plane_image)

                    # Display the processed image on the canvas
                    self.display_image_on_canvas(self.display_image_processed, self.canvas_processed)
                    
            else:
                                 # Display an error message if no image is loaded
                     messagebox.showinfo("Error", "No image loaded")
                    
# /////////////////////////////////// gray level////////////////////////////////
    def gray_level_slicing(self):
        
        if self.image is not None:
            
      
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

            # Apply gray-level slicing
            for i in range(gray_image.shape[0]):
                for j in range(gray_image.shape[1]):
                    if 130 < gray_image[i, j] < 200:
                        gray_image[i, j] = 255
                    else:
                        gray_image[i, j] = gray_image[i, j]

            # Convert the sliced image to Tkinter PhotoImage
            self.display_image_processed = self.convert_cv_image_to_tk_image(gray_image)

            # Display the processed image on the canvas
            self.display_image_on_canvas(self.display_image_processed, self.canvas_processed)
        else:
            
     # Display an error message if no image is loaded
            messagebox.showinfo("Error", "No image loaded")


            
    #   fliter
    # ////////////////////////////smothing //////////////////
    # /////// cone/////
    def apply_smoothing_filter(self):
        
        if self.image is not None:
            
        
        
            # Convert the BGR image to grayscale
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            # Create a kernel for smoothing 
            self.smoothing_kernel = np.ones((5, 5), np.float32) / 25

            # Apply the smoothing filter using filter2D
            smoothed_image = cv2.filter2D(gray_image, -1, self.smoothing_kernel)

            # Convert the smoothed image to Tkinter PhotoImage
            self.display_image_processed = self.convert_cv_image_to_tk_image(smoothed_image)

            # Display the processed image on the canvas
            self.display_image_on_canvas(self.display_image_processed, self.canvas_processed)
        else:
            # Display an error message if no image is loaded
            messagebox.showinfo("Error", "No image loaded")
        
        
        
        
        #  ////////////////////////////// median filter //////////////////////////////////////////////////s
    def apply_median_blur(self):
        if self.image is not None:
            

            # Apply median blur using the specified kernel size 
            ksize = 5  # kernel size
            blurred_image = cv2.medianBlur(self.image, ksize)

            # Convert the blurred image to Tkinter PhotoImage
            self.display_image_processed = self.convert_cv_image_to_tk_image(blurred_image)

            # Display the processed image on the canvas
            self.display_image_on_canvas(self.display_image_processed, self.canvas_processed)
        else:
            
              # Display an error message if no image is loaded
            messagebox.showinfo("Error", "No image loaded")

  


    # Apply sharpening filter
    # def apply_sharpening_filter(self):
    #     # Convert the BGR image to grayscale
    #     gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    #     # Calculate gradients in x and y directions using the Sobel operator
    #     grad_x = cv2.Sobel(gray_image.astype(np.float64), cv2.CV_64F, 1, 0, ksize=3)
    #     grad_y = cv2.Sobel(gray_image.astype(np.float64), cv2.CV_64F, 0, 1, ksize=3)

    #     # Combine gradients to get the sharpened image
    #     sharpened_image = cv2.addWeighted(gray_image, 1.5, grad_x, -0.5, 0)

    #     # Convert the sharpened image to Tkinter PhotoImage
    #     self.display_image_processed = self.convert_cv_image_to_tk_image(sharpened_image)

    #     # Display the processed image on the canvas
    #     self.display_image_on_canvas(self.display_image_processed, self.canvas_processed)
    
    
    
    # ///////////////////// apply   edge by finding sharp changes ///////////////////////////////////////////////
    # ////////////////////////  horizontal edge 
    def apply_h_edges(self):
        # Check if there is an image loaded
        if self.image is not None:
            # Define a horizontal edge detection kernel
            kernel_H = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
            
            # Apply the horizontal edge detection filter
            
            dst_H = cv2.filter2D(self.image, cv2.CV_16SC1, kernel_H)
            
            # Convert the result to a valid image format
            dst_H = cv2.convertScaleAbs(dst_H)

            # Convert the processed image to Tkinter format
            self.display_image_processed = self.convert_cv_image_to_tk_image(dst_H)
           # Display the processed image on the canvas
            self.display_image_on_canvas(self.display_image_processed, self.canvas_processed)
        else:
             # Display an error message if no image is loaded
            messagebox.showinfo("Error", "No image loaded")
            
            
            
            
            
# /////////////////////////////// vertical edge detectio
    def apply_v_edges(self):
        # Check if there is an image loaded
        if self.image is not None:
            # Define a vertical edge detection kernel
            kernel_V = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            
            # Apply the vertical edge detection filter
            dst_V = cv2.filter2D(self.image, cv2.CV_16SC1, kernel_V)
            
            # Convert the result to a valid image format
            dst_V = cv2.convertScaleAbs(dst_V)

            # Convert the processed image to Tkinter format
            self.display_image_processed = self.convert_cv_image_to_tk_image(dst_V)
            
            # Display the processed image on the canvas
            self.display_image_on_canvas(self.display_image_processed, self.canvas_processed)
        else:
            # Display an error message if no image is loaded
            messagebox.showinfo("Error", "No image loaded")
            
            
    def apply_thresholding(self):
        # Check if an image is loaded
        if self.image is not None:
            # Choose a threshold value (you can modify this)
            threshold_value = 128

            # Convert the image to grayscale if it's a color image
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

            # Apply thresholding
            _, thresholded_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

          # Convert the thresholded image to Tkinter PhotoImage
            self.display_image_processed = self.convert_cv_image_to_tk_image(thresholded_image)

            # Display the processed image on the canvas
            self.display_image_on_canvas(self.display_image_processed, self.canvas_processed)
        else:
            # Display an error message if no image is loaded
            messagebox.showinfo("Error", "No image loaded")

            
            
            
            
            

            
            
# ////////////////////////////////apply_frequency_domain_enhancement to remove spark noise and get less blurring///////////////////////////////
    def apply_dft(self):
        # Check if an image is loaded
        if self.image is not None:
            I = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            # Step 1: Image Setup
            m = cv2.getOptimalDFTSize(I.shape[0])
            n = cv2.getOptimalDFTSize(I.shape[1])
            padded = cv2.copyMakeBorder(I, 0, m - I.shape[0], 0, n - I.shape[1], 0, 0)
            padded = padded.astype(np.float32) / 255.0
            print("test")

            # Step 2: Make place for both the complex and the real values
            planes = [padded, np.zeros(padded.shape, np.float32)]
            complexI = cv2.merge(planes)
            print("t")
            # print(padded.dtype)

           # Step 3: Apply DFT
            complexI = cv2.dft(complexI)
            print("tt")
        # Step 4: Rearrange frequencies
            cx, cy = complexI.shape[1] // 2, complexI.shape[0] // 2
            p1 = complexI[0:cy, 0:cx]
            p2 = complexI[0:cy, cx:]
            p3 = complexI[cy:, 0:cx]
            p4 = complexI[cy:, cx:]

            temp = p1.copy()
            p1[:,:] = p4
            p4[:,:] = temp

            temp = p2.copy()
            p2[:,:] = p3
            p3[:,:] = temp
            print("th")
        # Step 5: Filter
            d0 = 50
            Lfilter = np.zeros((complexI.shape[0], complexI.shape[1]), np.float32)
            
            for i in range(Lfilter.shape[0]):
                for j in range(Lfilter.shape[1]):
                    z1 = i - Lfilter.shape[0] // 2
                    z2 = j - Lfilter.shape[1] // 2
                    if np.sqrt(z1**2 + z2**2) <= d0:
                        Lfilter[i, j] = 1
                    else:
                        Lfilter[i, j] = 0
            print("j")

            # Step 6: Apply filter
            planes = cv2.split(complexI)
            outR = planes[0] * Lfilter
            outI = planes[1] * Lfilter
            out_complexI = cv2.merge([outR, outI])
            print("testh")

            # Step 7: IDFT
            out_complexI = cv2.idft(out_complexI)
            planes = cv2.split(out_complexI)
            print("testh2")



            # Step 8: Calculate magnitude and normalize image
            out = cv2.magnitude(planes[0], planes[1])
            out = cv2.normalize(out, None, 0, 1, cv2.NORM_MINMAX)

            print("test55")
            
             # Step 9: Display result
            cv2.imshow("after IDFT", out)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            # Convert the thresholded image to Tkinter PhotoImage
            self.display_image_processed = self.convert_cv_image_to_tk_image(out)
            print("testh58")


            #  Display the processed image on the canvas
        
            self.display_image_on_canvas(self.display_image_processed, self.canvas_processed)
            print("testhk")
     
   
        else:
            messagebox.showinfo("Error", "No image loaded")








# /////////////////////////////// object detection ////////////////////////////////////////////////////////////
    def object_detection_image(self):
        
        if self.image is not None:
    

  
            # Read the image
            img = self.image

            # Load class names from a file
            classnames = []
            classfile = 'files/thing.names'

            with open(classfile, 'rt') as f:
                classnames = f.read().rstrip('\n').split('\n')

            # Load pre-trained model
            p = 'files/frozen_inference_graph.pb'
            v = 'files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'

            net = cv2.dnn_DetectionModel(p, v)
            net.setInputSize(320, 230)
            net.setInputScale(1.0 / 127.5)
            net.setInputMean((127.5, 127.5, 127.5))
            net.setInputSwapRB(True)

            # Detect objects in the image
            classIds, confs, bbox = net.detect(img, confThreshold=0.5)

            # Draw bounding boxes and labels
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                cv2.rectangle(img, box, color=(0, 255, 0), thickness=3)
                cv2.putText(img, classnames[classId - 1], (box[0] + 10, box[1] + 20),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), thickness=2)
            # Display the result
            # cv2.imshow('Object Detection', img)
            # cv2.waitKey(0)
            # Display the image with detections
            self.display_image_processed = self.convert_cv_image_to_tk_image(img)
            self.display_image_on_canvas(self.display_image_processed, self.canvas_processed)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.show()
        else:
            messagebox.showinfo("Error", "No image loaded")
            
    def object_detection_video(self):
        if self.image is not None:
        
            # Load class names from a file
            classnames = []
            classfile = 'files/thing.names'

            with open(classfile, 'rt') as f:
                classnames = f.read().rstrip('\n').split('\n')

            # Load pre-trained model
            p = 'files/frozen_inference_graph.pb'
            v = 'files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'

            net = cv2.dnn_DetectionModel(p, v)

            # Set input size, scale, mean, and color swapping
            net.setInputSize(320, 230)
            net.setInputScale(1.0 / 127.5)
            net.setInputMean((127.5, 127.5, 127.5))
            net.setInputSwapRB(True)

            # Open webcam
            cap = cv2.VideoCapture(0)  # 0 corresponds to the default camera, change if you have multiple cameras

            while True:
                ret, frame = cap.read()

                # Detect objects in the frame
                classIds, confs, bbox = net.detect(frame, confThreshold=0.5)

                if len(classIds) != 0:
                    for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                        cv2.rectangle(frame, box, color=(0, 255, 0), thickness=3)
                        cv2.putText(frame, classnames[classId - 1], (box[0] + 10, box[1] + 20),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), thickness=2)

                # Display the result
                cv2.imshow('Video Object Detection', frame)
                # plt.imshow(cv2.cvtColor(frame ,cv2.COLOR_BGR2RGB))
                # plt.show()

                # Exit when 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Release webcam and close all windows
            cap.release()
            cv2.destroyAllWindows()
        else:
                         # Display an error message if no image is loaded
            messagebox.showinfo("Error", "No image loaded")
            
# /////////////////////////////// OCR /////////////////////////////////////////////////////

    def apply_ocr(self):
        if self.image is not None:
                    # Read the image
            img = self.image

            # Convert the image to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Apply bilateral filter for noise reduction and smoothing
            bfilter = cv2.bilateralFilter(gray, 11, 17, 17)

            # Apply Canny edge detection
            edged = cv2.Canny(bfilter, 30, 200)

            # Find contours in the edged image
            keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contourss = imutils.grab_contours(keypoints)
            contours = sorted(contourss, key=cv2.contourArea, reverse=True)[:10]

            # Initialize variable for OCR location
            location = None

            # Iterate through contours to find the one with four corners (approxPolyDP)
            for contour in contours:
                approx = cv2.approxPolyDP(contour, 10, True)
                if len(approx) == 4:
                    location = approx
                    break

            # Print the location of the contour (four corners)
            print(location)

            # Create an empty black mask with the same shape as the grayscale image
            mask = np.zeros(gray.shape, np.uint8)

            # Draw the contour (location) on the mask, filling it with white color
            new_image = cv2.drawContours(mask, [location], -1, 255, -1)

            # Perform bitwise AND operation to isolate the region of interest in the original image
            new_image_1 = cv2.bitwise_and(img, img, mask=mask)

            # Find the coordinates of all white pixels in the mask
            (x, y) = np.where(mask == 255)

            # Find the minimum and maximum coordinates to define the bounding box
            (x1, y1) = (np.min(x), np.min(y))
            (x2, y2) = (np.max(x), np.max(y))

            # Crop the image based on the contour
            cropped_image = gray[x1:x2 + 1, y1:y2 + 1]

            # Display the cropped image
            plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

            # Initialize EasyOCR Reader for English language
            reader = easyocr.Reader(['en'])

            # Perform OCR on the cropped image
            result = reader.readtext(cropped_image)
            print(result)
            # r_label= tk.Label(self.frame_B_od,text=result)
            # r_label.place(x=400,y=450)
            

            # Extract OCR text and font information
            text = result[0][-2]
            font = cv2.FONT_HERSHEY_SIMPLEX

            # Draw the OCR text on the original image
            res = cv2.putText(img, text=text, org=(location[0][0][0], location[1][0][1] + 60), fontFace=font,
                            fontScale=1, color=(0, 255, 0), thickness=3, lineType=cv2.LINE_AA)
            res = cv2.rectangle(img, tuple(location[0][0]), tuple(location[2][0]), (0, 255, 0), 3)
            
                                    # Convert the thresholded image to Tkinter PhotoImage
            self.display_image_processed = self.convert_cv_image_to_tk_image(res)
            print("testh58")


            #  Display the processed image on the canvas
        
            self.display_image_on_canvas(self.display_image_processed, self.canvas_processed)
            print("testhk")

            # # Display the result image with OCR text and bounding box
            # plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
            # plt.show()
            

            
            
        else:
                        # Display an error message if no image is loaded
            messagebox.showinfo("Error", "No image loaded")






                        
                    




























            
            
            
            
            
            
        
        

# Create the main window
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingToolbox(root)
    root.mainloop()
