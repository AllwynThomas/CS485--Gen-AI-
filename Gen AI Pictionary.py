import sys
import os
import pygame
import pygame_gui
import pygame_gui.elements.ui_button
import ctypes
import random
import time
import quickdraw
import pandas as pd
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
#tf.config.set_visible_devices([], 'GPU')                    # Does not use GPU for processing
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from PIL import Image, ImageOps, ImageEnhance
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from math import ceil

# Checking if TFlite model exists
CNNmodelExists = False
GANmodelExists = False
if Path("models/CNN/tflite_model_40.tflite").exists():
    CNNmodelExists = True
if Path("models/GAN").exists():
    GANmodelExists = True

# Getting QuickDraw Images from API
GAN_image_size = 28
CNN_image_size = 40

def generate_class_images(name, max_drawings, recognized, img_size, path):
    directory = Path(path + name)
    if directory.exists():
        return
    
    directory.mkdir(parents=True)
    try:
        images = quickdraw.QuickDrawDataGroup(name, max_drawings=max_drawings, recognized=recognized, cache_dir="data/.quickdrawcache")
        for img in images.drawings:
            filename = directory.as_posix() + "/" + str(img.key_id) + ".png"
            img.get_image(stroke_width=3).resize((img_size, img_size)).save(filename)
    except:
        return

def download_images_parallel(labels, max_drawings, recognized, img_size, path):
    # Use the number of CPUs or threads available
    max_threads = os.cpu_count() or 4  # Fallback to 4 if os.cpu_count() returns None
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = [
            executor.submit(generate_class_images, label, max_drawings, recognized, img_size, path)
            for label in labels
        ]
        for future in futures:
            future.result()

if __name__ == "__main__":
    categories = quickdraw.QuickDrawData().drawing_names
    #start = time.time()
    if (not CNNmodelExists):
        download_images_parallel(categories, max_drawings=3000, recognized=True, img_size=CNN_image_size, path="data/quickdraw_images_40/")
    if (not GANmodelExists):
        download_images_parallel(categories, max_drawings=3000, recognized=True, img_size=GAN_image_size, path="data/quickdraw_images_28/")    
    #end = time.time()
    #print(f"Image Download Time: {(end-start)//60}m {(end-start)%60}s")

# Picking Categories
player_random_categories = random.sample(categories, 3)
dir_path = f"{os.getcwd()}/figures/GAN/RMSProp_1000_Epochs_28"
all_items = os.listdir(dir_path)
dirs = [item for item in all_items if os.path.isdir(os.path.join(dir_path, item))]
ai_random_categories = random.sample(dirs, 3)
player_categories = {}
ai_categories = {}
ai_guesses = {}
for r in range(1, 4):
    player_categories[r] = player_random_categories[r-1]
    ai_categories[r] = ai_random_categories[r-1]
#print(player_categories, ai_categories)

# Function for GAN to draw and save images
def GANdraw(model_label, roundNum, num):
    if (num < 7):
        rand_num = random.randint(4, 9)
        image_idx = (num + 4) * 100
        saved_image = Image.open(f"figures/GAN/RMSProp_1000_Epochs_28/{model_label}/{model_label}_Epoch_{image_idx}/{rand_num}.png").resize((700, 700), resample=Image.Resampling.LANCZOS)
        contrastEnhancer = ImageEnhance.Contrast(saved_image)
        contrastImg = contrastEnhancer.enhance(1.5)
        sharpnessEnhancer = ImageEnhance.Sharpness(contrastImg)
        sharpnessEnhancer.enhance(1.5).save(f"images/temp/ai_temp/round_{roundNum}/{num}.png")
    else:        
        # Load TFLite model and allocate tensors
        interpreter = tf.lite.Interpreter(model_path=f"models/GAN/{model_label}_RMSProp_1000_Epoch_28_model_lite.tflite")
        interpreter.allocate_tensors()

        # Get input and output tensors
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        input_data = tf.convert_to_tensor(np.random.uniform(-1.0, 1.0, size=[1, 100]), dtype=tf.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        # Saving generated image
        output_data = interpreter.get_tensor(output_details[0]['index']) #kerasModel.predict(input_data)
        img_array = output_data[0, :, :, 0]  
        img_array = (img_array * 255).astype(np.uint8)
        img = Image.fromarray(img_array, mode="L").resize((700, 700), resample=Image.Resampling.LANCZOS)
        contrastEnhancer = ImageEnhance.Contrast(img)
        contrastImg = contrastEnhancer.enhance(1.5)
        sharpnessEnhancer = ImageEnhance.Sharpness(contrastImg)
        sharpnessEnhancer.enhance(1.5).save(f"images/temp/ai_temp/round_{roundNum}/{num}.png")

# Function to delete temp images every round
def tempDelete(roundNum):
    tempdir = f"{os.getcwd()}/images/temp/ai_temp/round_{roundNum}"
    files = os.listdir(tempdir)
    for file in files:
        try:
            file_path = os.path.join(tempdir, file)
            os.remove(file_path)
        except Exception:
            return
    
# Loading & Generating AI drawings for the 3 rounds
for roundNum, label in ai_categories.items():
    for i in range(10):
        GANdraw(label, roundNum, i)

# Training GAN for each category
for model_category in categories:
    constantNoise = np.random.uniform(-1.0, 1.0, 
                                        size=[10, 100])
    # Getting dataset for training GAN
    if (not GANmodelExists):
        # Loading images
        model_category = model_category
        dataset_dir = Path(f"data/quickdraw_images_28/{model_category}")
        GAN_train_ds = tf.keras.utils.image_dataset_from_directory(
            dataset_dir,
            seed=42,
            color_mode="grayscale",
            image_size=(GAN_image_size, GAN_image_size),
            batch_size=40,
            labels=None,
            label_mode=None
        )
        
        # Normalizing Images
        def GANpreprocess(image):
            image = tf.cast(image, tf.float32) / 255.0
            return image
        AUTOTUNE = tf.data.AUTOTUNE
        GAN_train_ds = GAN_train_ds.map(GANpreprocess, num_parallel_calls=AUTOTUNE)
        GAN_train_ds = GAN_train_ds.cache().shuffle(1024).prefetch(buffer_size=AUTOTUNE)

    # Creating GAN Model
    if (not GANmodelExists):
        # Building Generator
        def build_generator(noise_dim=100, depth=64, p=0.4):
            model = Sequential([
                # Input layer and first dense layer
                keras.layers.InputLayer(input_shape=(noise_dim,)),
                keras.layers.Dense(7 * 7 * depth),
                keras.layers.BatchNormalization(momentum=0.9),
                keras.layers.Activation('relu'),
                keras.layers.Reshape((7, 7, depth)),
                keras.layers.Dropout(p),

                # First UpSampling and Conv2DTranspose
                keras.layers.UpSampling2D(),
                keras.layers.Conv2DTranspose(
                    int(depth / 2), kernel_size=5, padding='same'),
                keras.layers.BatchNormalization(momentum=0.9),
                keras.layers.Activation('relu'),

                # Second UpSampling and Conv2DTranspose
                keras.layers.UpSampling2D(),
                keras.layers.Conv2DTranspose(
                    int(depth / 4), kernel_size=5, padding='same'),
                keras.layers.BatchNormalization(momentum=0.9),
                keras.layers.Activation('relu'),

                # Third Conv2DTranspose
                keras.layers.Conv2DTranspose(
                    int(depth / 8), kernel_size=5, padding='same'),
                keras.layers.BatchNormalization(momentum=0.9),
                keras.layers.Activation('relu'),

                # Output layer
                keras.layers.Conv2D(1, kernel_size=5, padding='same', activation='sigmoid')
            ])
            return model
        generator = build_generator()
        #generator.summary()

        # Building Discriminator
        def build_discriminator(depth=64, p=0.4):
            model = Sequential([
                # Input layer
                keras.layers.InputLayer(input_shape=(GAN_image_size, GAN_image_size, 1)),

                # First convolutional layer
                keras.layers.Conv2D(depth * 1, kernel_size=5, strides=2, padding='same', activation='relu'),
                keras.layers.Dropout(p),

                # Second convolutional layer
                keras.layers.Conv2D(depth * 2, kernel_size=5, strides=2, padding='same', activation='relu'),
                keras.layers.Dropout(p),

                # Third convolutional layer
                keras.layers.Conv2D(depth * 4, kernel_size=5, strides=2, padding='same', activation='relu'),
                keras.layers.Dropout(p),

                # Fourth convolutional layer
                keras.layers.Conv2D(depth * 8, kernel_size=5, strides=1, padding='same', activation='relu'),
                keras.layers.Dropout(p),

                # Flatten and dense output layer
                keras.layers.Flatten(),
                keras.layers.Dense(1, activation='sigmoid')
            ])
            return model
        discriminator = build_discriminator()
        #discriminator.summary()
        
        discriminator.compile(
            loss='binary_crossentropy', 
            optimizer=keras.optimizers.RMSprop(learning_rate=0.0008, clipvalue=1.0), 
            metrics=['accuracy']
        )
        
        # Creating Adversarial Model
        # Generator inputs
        z = keras.layers.Input(shape=(100,))                

        # Generator output
        img = generator(z)             

        # Discriminator input
        discriminator.trainable = False
        pred = discriminator(img)

        # Adversarial model
        adversarial_model = keras.Model(z, pred)
        adversarial_model.compile(
            loss='binary_crossentropy',
            optimizer=keras.optimizers.RMSprop(learning_rate=0.0004, clipvalue=1.0),
            metrics=['accuracy']
        )
        
        # Training
        def train_gan(epochs=1000, batch=40, z_dim=100):
            d_metrics = []
            a_metrics = []
            
            running_d_loss = 0
            running_d_acc = 0
            running_a_loss = 0
            running_a_acc = 0
        
            for i in range(epochs):
                # Sample real images
                for image_batch in GAN_train_ds.take(1):  # Take a single batch
                    real_imgs = np.array(image_batch)  # Convert images to NumPy array
                
                # generate fake images: 
                fake_imgs = generator.predict(
                    np.random.uniform(-1.0, 1.0, 
                                    size=[batch, z_dim]), verbose=0)
                
                # concatenate images as discriminator inputs:
                x = np.concatenate((real_imgs,fake_imgs))
                
                # Assign y labels for discriminator
                y = np.ones([2 * batch, 1])
                y[batch:] = 0  # Fake labels as 0
                
                # train discriminator: 
                d_metrics.append(
                    discriminator.train_on_batch(x,y)
                )
                running_d_loss += d_metrics[-1][0]
                running_d_acc += d_metrics[-1][1]
                
                # adversarial net's noise input and "real" y: 
                noise = np.random.uniform(-1.0, 1.0, 
                                        size=[batch, z_dim])
                y = np.ones([batch,1])
                
                # train adversarial net: 
                a_metrics.append(
                    adversarial_model.train_on_batch(noise,y)
                ) 
                running_a_loss += a_metrics[-1][0]
                running_a_acc += a_metrics[-1][1]
                
                # Periodically print progress & generate images
                if (i + 1) % 100 == 0:
                    print(f'Epoch #{i+1}')
                    log_msg = f"{i+1}: [D loss: {running_d_loss/(i+1):.4f}, acc: {running_d_acc/(i+1):.4f}]"
                    log_msg += f"  [A loss: {running_a_loss/(i+1):.4f}, acc: {running_a_acc/(i+1):.4f}]"
                    print(log_msg)
                    
                    try:
                        Path(f"figures/GAN/RMSProp_1000_Epochs_28/{model_category}/{model_category}_Epoch_{i+1}").mkdir(parents=True)
                    except FileExistsError:
                        pass
                    # Generate and plot some images for visualization and save others for later
                    gen_imgs = generator.predict(constantNoise, verbose=0)
                    for idx in range(4, 10):  
                        img_array = gen_imgs[idx, :, :, 0]  
                        img_array = (img_array * 255).astype(np.uint8)  
                        img = Image.fromarray(img_array, mode="L")
                        img.save(f"figures/GAN/RMSProp_1000_Epochs_28/{model_category}/{model_category}_Epoch_{i+1}/{idx}.png")
                    plt.figure(figsize=(6, 6))
                    for k in range(4):
                        plt.subplot(2, 2, k+1)
                        plt.imshow(gen_imgs[k, :, :, 0], cmap='gray')
                        plt.axis('off')
                    plt.suptitle(f"{model_category.capitalize()} at Epoch {i+1}\n{log_msg}")
                    plt.tight_layout()
                    plt.savefig(f"figures/GAN/RMSProp_1000_Epochs_28/{model_category}/{model_category}_Epoch_{i+1}/visual.png")
                    plt.close()
            return a_metrics, d_metrics
        a_metrics_complete, d_metrics_complete = train_gan()
        
        # Saving the model
        generator.export(f"./models/GAN/{model_category}_RMSProp_1000_Epoch_28_model_lite")
        
        # Converting to Tflite
        converter = tf.lite.TFLiteConverter.from_saved_model(f"models/GAN/{model_category}_RMSProp_1000_Epoch_28_model_lite")
        tflite_model = converter.convert()

        # Saving the Tflite Model.
        with open(f"models/GAN/{model_category}_RMSProp_1000_Epoch_28_model_lite.tflite", 'wb') as f:
            f.write(tflite_model)

if (not CNNmodelExists or not Path("data/categories.txt").exists()):
    # Splitting Dataset into Train/Validate Sets    
    dataset_dir = Path("data/quickdraw_images_40")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        color_mode="grayscale",
        image_size=(CNN_image_size, CNN_image_size),
        batch_size=32
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        color_mode="grayscale",
        image_size=(CNN_image_size, CNN_image_size),
        batch_size=32
    )

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(2000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    categories = train_ds.class_names
    if (not Path("data/categories.txt").exists()):
        with open("data/categories.txt", "w") as f:
            f.write("\n".join(categories))

# Load correctly ordered categories for predictions
with open("data/categories.txt", "r") as f:
    categories = f.read().splitlines()

if (not CNNmodelExists):
    # Building CNN Model
    n_classes = 345
    input_shape = (40, 40, 1)

    model = Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.Rescaling(1./255),
        layers.BatchNormalization(),

        layers.Conv2D(6, kernel_size=3, padding="same", activation="relu"),
        layers.Conv2D(8, kernel_size=3, padding="same", activation="relu"),
        layers.Conv2D(10, kernel_size=3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=2),
        
        layers.Flatten(),
        
        layers.Dense(700, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(500, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(400, activation="relu"),
        layers.Dropout(0.2),

        layers.Dense(n_classes, activation="softmax")
    ])

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'])

    model.summary()

    # Training and Saving CNN Base Model
    epochs = 10

    #logdir = os.path.join("models/CNN/logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    #tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=1,
        #callbacks=[tensorboard_callback]
    )
    model.save("./models/CNN/bigger_model_40")

    # Converting the Model
    model.export('./models/CNN/bigger_model_40_lite')
    converter = tf.lite.TFLiteConverter.from_saved_model("models/CNN/bigger_model_40_lite")
    tflite_model = converter.convert()

    # Saving the Tflite Model.
    with open('models/CNN/tflite_model_40.tflite', 'wb') as f:
        f.write(tflite_model)

# Function for preprocessing player drawings to make them similar to training data
def imagePreprocessing(image):
    # Converting to grayscale
    img = image.convert('L')
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Find bounding box of non-white pixels
    non_white_pixels = np.argwhere(img_array < 255)
    try:
        y_min, x_min = non_white_pixels.min(axis=0)
        y_max, x_max = non_white_pixels.max(axis=0)
    except:
        # Return white image if no drawing and +1 to aiPoints for player not drawing
        global aiPoints
        aiPoints += 1
        return Image.new('L', (CNN_image_size, CNN_image_size), color=255)    
    
    # Crop to bounding box
    cropped_img = img_array[y_min:y_max + 1, x_min:x_max + 1]
    
    # Resize to fit within output_size while maintaining aspect ratio
    cropped_img_pil = Image.fromarray(cropped_img)
    cropped_img_resized = ImageOps.fit(cropped_img_pil, (CNN_image_size, CNN_image_size), method=Image.Resampling.LANCZOS)
    
    # Pad to ensure it's centered on a 40x40 canvas
    padded_img = Image.new('L', (CNN_image_size, CNN_image_size), color=255)  # Create white canvas
    padded_img.paste(cropped_img_resized, (0, 0))  # Paste resized image
    
    # Final resizing to ensure size compatibility
    final_img = padded_img.resize((CNN_image_size, CNN_image_size))
    return final_img

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="models/CNN/tflite_model_40.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function for CNN to guess player's drawing
def CNNguess(img_name):
    global ai_guesses
    # Preprocessing player's drawing
    player_img = Image.open(f"images/temp/player_temp/{img_name}.png")
    processed_img = imagePreprocessing(player_img)
    #processed_img.save(f"images/temp/test/{img_name}.png")

    input_data = tf.convert_to_tensor(np.array(processed_img, dtype=np.float32))
    input_data = np.expand_dims(input_data, axis=(0, -1))
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    # Finding top 3 possible categories based on 5 highest probabilities
    output_data = interpreter.get_tensor(output_details[0]['index'])[0] #kerasModel.predict(input_data)[0]
    top5_categories = np.argpartition(output_data, -5)[-5:]
    top5_categories_sorted = top5_categories[np.argsort(output_data[top5_categories])[:]]
    for i in top5_categories_sorted:
        ai_guesses[categories[i]] = round(output_data[i]*100, 2)
    #print(f"\nPredicted Categories: {ai_guesses}\n")
    
    # Deleting temp drawing
    current_dir = os.getcwd()
    file_name = f"images/temp/player_temp/{img_name}.png"
    file_path = os.path.join(current_dir, file_name)
    try:
        os.remove(file_path)
    except Exception:
        return

# Pygame Config
ctypes.windll.shcore.SetProcessDpiAwareness(True)                   # Sharper Window
pygame.init()
fps = 0
fpsClock = pygame.time.Clock()
width, height = 640, 480
screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
font = pygame.font.SysFont('Helvetica', 30)
ui_manager = pygame_gui.UIManager((1080, 1080), "themes/button_themes.json")

# Variables
buttons = {}
colorMappings = {"#1,1": "#FFFFFF", "#1,2": "#C1C1C1", "#1,3": "#EF130B", "#1,4": "#FF7100", "#1,5": "#FFE400", "#1,6": "#00CC00", "#1,7": "#00FF91", "#1,8": "#00B2FF", "#1,9": "#231FD3", "#1,10": "#A300BA", "#1,11": "#DF69A7", "#1,12": "#FFAC8E", "#1,13": "#A0522D",
                 "#2,1": "#000000", "#2,2": "#505050", "#2,3": "#740B07", "#2,4": "#C23800", "#2,5": "#E8A200", "#2,6": "#004619", "#2,7": "#00785D", "#2,8": "#00569E", "#2,9": "#0E0865", "#2,10": "#550069", "#2,11": "#873554", "#2,12": "#CC774D", "#2,13": "#63300D"}
drawColor = [0,0,0]
brushSize = 3
canvasSize = [800, 800]
playerTurn = True               # 1 = Player; 0 = AI Opponent
roundNum = 1
gameOver = False
hintSet = False
correct = False
playerPoints = 0
aiPoints = 0

# Round Timer
roundTime = 60
counter, timerText = roundTime, str(roundTime).rjust(3)
TIMEREVENT = pygame.USEREVENT+3
pygame.time.set_timer(TIMEREVENT, 1000)
timerFont = pygame.font.SysFont('Courier New', 30)
hintFont = pygame.font.SysFont('Consolas', 30)

# Canvas Setup
canvas = pygame.Surface(canvasSize)
canvas.fill((255, 255, 255))

# Text Input
textInputRect = pygame.Rect(1210, 798, 270, 40)
enteredTextRect = pygame.Rect(1210, 798-40, 270, 40)
userText = "Type your guess here..."
enteredText = ""
inputActive = pygame.Color('white')
inputPassive = pygame.Color('#B8FF9B')
inputColor = inputPassive
inputTextColor = pygame.Color('darkgray')
isTextInputActive = False
enteredTextColor = pygame.Color('lightskyblue')

# Color Swatches (Buttons)                                                       
button_row_width = 50
button_row_height = 50
start_x = 200
start_y = 110
spacing = 5
for i in range(1, 3):
    for j in range(1, 14):
        position =  (start_x + i * spacing + ((i - 1) * button_row_width),
                    (start_y + j * spacing + ((j - 1) * button_row_height)))
        button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect( position,
                                                                (button_row_width,
                                                                button_row_height)),
                                                                text='',
                                                                manager=ui_manager,
                                                                object_id='#' + str(i) + ',' + str(j))
        buttons[str(button.object_ids)] = button

# Handler Functions
def changeColor(color):
    global drawColor
    drawColor = color
        
def save(name, playerType):
    pygame.image.save(canvas, f"images/{playerType}/{name}.png")

# Main Game Loop
while True:
    screen.fill((200,200,200))
    time_delta = fpsClock.tick(fps) / 1000
    for event in pygame.event.get():
        if (event.type == pygame.QUIT):
            pygame.quit()
            sys.exit()
        if (playerTurn and event.type == pygame_gui.UI_BUTTON_PRESSED):
            buttonID = str(event.ui_element.object_ids)[2:-2]
            if (buttonID in colorMappings):
                changeColor(colorMappings[buttonID])
        if (event.type == pygame.MOUSEBUTTONDOWN):
            if (textInputRect.collidepoint(event.pos)):
                isTextInputActive = True
            else:
                isTextInputActive = False
        if (event.type == pygame.KEYDOWN and isTextInputActive):
            if (event.key == pygame.K_BACKSPACE):
                userText = userText[:-1]
            elif (event.key == pygame.K_RETURN):
                enteredText = userText
                userText = ""
            else:
                userText += event.unicode
        if (event.type == TIMEREVENT):
            if (roundNum == 1 and playerTurn and counter == 60):
                targetWord = player_categories[roundNum]
            counter -= 1
            timerText = str(counter).rjust(3)
            # Decrementing timer
            if (counter > 0):
                timerText = str(counter).rjust(3)
                # Displaying AI's word as blanks for the player to guess
                if (not playerTurn):
                    if (((60-counter) % secondsPerLetter) == 0):
                        underscore_indices = [i for i, char in enumerate(wordHint) if char == "_"]
                        randIdx = random.choice(underscore_indices)
                        wordHint = wordHint[:randIdx] + targetWord[randIdx] + wordHint[randIdx+1:]
                # CNN guessing player's image every 10s
                if (playerTurn and 60-counter > 0 and (counter % 60) % 10 == 0):
                    imgName = f"{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())}_round_{roundNum}_{60-counter}s_player_drawing"
                    save(imgName, "temp/player_temp")
                    CNNguess(imgName)
                # Entering CNN guesses in chat every 2s
                if (playerTurn and 60-counter > 10 and (counter % 60 - 10) % 2 == 0):
                    guess = ai_guesses.popitem()
                    enteredText = guess[0]
                    # Adding points based on how quick the guess was and how good the drawing is
                    if (enteredText == targetWord and not correct):
                        correct = True
                        aiPoints += ceil(counter / 12)
                        if (guess[1] <= 1):
                            aiPoints += 2
                        elif (guess[1] <= 25):
                            aiPoints += 1
                        elif (guess[1] >= 50):
                            playerPoints += 1
                        elif (guess[1] >= 90):
                            playerPoints += 2   
                # GAN drawing image every 6s
                if (not playerTurn and (counter % 6) == 0):
                    idx = round((60-counter) / 6)
                    savedImg = pygame.image.load(f"images/temp/ai_temp/round_{roundNum}/{idx}.png")
                    canvas.blit(savedImg, ((canvas.width - savedImg.width) / 2, (canvas.height - savedImg.height) / 2))
                    pygame.display.flip()
            # Switching between player/ai turns when timer hits 0 and incrementing rounds
            elif ((counter <= 0) and not(roundNum == 3 and not playerTurn)):
                playerTurn = not playerTurn
                correct = False
                ai_guesses = {}
                if (playerTurn == True):
                    timerText = f"Player's Turn!\nAI Drawn Word: {targetWord}"
                    save(f"{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())}_round_{roundNum}_ai_drawing", "ai")
                    tempDelete(roundNum)
                    roundNum += 1
                    hintSet = False
                else:
                    timerText = f"AI's Turn!\nPlayer Drawn Word: {targetWord}"
                    save(f"{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())}_round_{roundNum}_player_drawing", "player")
                drawColor = "#000000"
                screen.blit(timerFont.render(timerText, True, (0, 0, 0)), (32, 48))
                canvas.fill("#FFFFFF")
                pygame.display.flip()
                time.sleep(3)
                counter = roundTime + 3
            # Game Over screen with total points and winner
            else:
                gameOver = True
                if (aiPoints > playerPoints):
                    winner = "AI Wins!"
                elif (playerPoints > aiPoints):
                    winner = "Player Wins!"
                else:
                    winner = "Tie Game!"
                timerText = f"Game Over!\n\nAI's Score: {aiPoints}\nPlayer's Score: {playerPoints}\n\n{winner}"
                screen.blit(timerFont.render(timerText, True, (0, 0, 0)), (32, 48))
                pygame.display.flip()
                tempDelete(roundNum)
        ui_manager.process_events(event)
    ui_manager.update(time_delta)
    
    if (not gameOver):
    
        x, y= screen.get_size()
        screen.blit(canvas, [x/2 - canvasSize[0]/2, y/2 - canvasSize[1]/2])
            
        # Drawing
        if (playerTurn and pygame.mouse.get_pressed()[0]):
            mx, my = pygame.mouse.get_pos()
            dx = mx - x/2 + canvasSize[0]/2
            dy = my - y/2 + canvasSize[1]/2
            pygame.draw.circle(
                canvas,
                drawColor,
                [dx, dy],
                brushSize,
            )
        
        # Displaying Target Words and Points
        if (playerTurn):
            # Displaying Player's word when drawing
            targetWord = player_categories[roundNum]
            if (correct):
                screen.blit(hintFont.render(targetWord, True, (50,205,50)), (1210, 35))
            else:
                screen.blit(hintFont.render(targetWord, True, (0,0,0)), (1210, 35))
        else:
            # Displaying AI's word with hints when drawing
            targetWord = ai_categories[roundNum]
            secondsPerLetter = round(60 / (len(targetWord)/2))
            if (not hintSet):
                wordHint = "_"*len(targetWord)
                space_indices = [i for i, char in enumerate(targetWord) if char == " "]
                for idx in space_indices:
                    wordHint = wordHint[:idx] + " " + wordHint[idx+1:]
                hintSet = True
            # Calculating player points based on how quickly they guess
            if (enteredText == targetWord and not correct):
                correct = True
                playerPoints += ceil(counter / 12)
            if (correct):
                screen.blit(hintFont.render(targetWord, True, (50,205,50)), (1210, 35))
            else:
                screen.blit(hintFont.render(f"{wordHint} ({len(wordHint)})", True, (0, 0, 0)), (1210, 35))
        screen.blit(timerFont.render(f"Player's Points: {playerPoints}", True, (0, 0, 0)), (1210, 80))
        screen.blit(timerFont.render(f"AI's Points: {aiPoints}", True, (0, 0, 0)), (1210, 110))
        
        # Text Input
        if (isTextInputActive):
            inputColor = inputActive
            inputTextColor = pygame.Color('black')
            if (not userText or userText == "Type your guess here..."):
                userText = ""
        else:
            inputColor = inputPassive
            if (not userText):
                userText = "Type your guess here..."
                inputTextColor = pygame.Color('darkgray')
        pygame.draw.rect(screen, inputColor, textInputRect)
        textSurface = font.render(userText, True, inputTextColor)
        screen.blit(textSurface, (textInputRect.x+5, textInputRect.y))
        textInputRect.w = max(270, textSurface.get_width()+10)
        
        # Entered Text
        pygame.draw.rect(screen, enteredTextColor, enteredTextRect)
        enteredTextSurface = font.render(enteredText, True, (0, 0, 0))
        screen.blit(enteredTextSurface, (enteredTextRect.x+5, enteredTextRect.y))
        enteredTextRect.w = max(270, enteredTextSurface.get_width()+10) 
            
        # Color Indicator
        pygame.draw.rect(
            screen,
            drawColor,
            pygame.Rect(235, 40, 50, 50),
            border_radius=5
        )
        pygame.draw.rect(
            screen,
            "#000000",
            pygame.Rect(235, 40, 50, 50),
            width=2,
            border_radius=5
        )
        ui_manager.draw_ui(screen)
        
    # Timer
    screen.blit(timerFont.render(timerText, True, (0, 0, 0)), (32, 48))
        
    pygame.display.flip()
    fpsClock.tick(fps)