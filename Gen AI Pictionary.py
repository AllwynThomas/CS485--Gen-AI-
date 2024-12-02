import sys
import os
import pygame
import pygame_gui
import pygame_gui.elements.ui_button
import ctypes
import time, datetime
import random
import quickdraw
import pandas as pd
import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')                    # Does not use GPU for processing
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from PIL import Image
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Checking if TFlite model exists
modelExists = False
if Path("models/CNN/tflite_model.tflite").exists():
    modelExists = True

# Getting QuickDraw Images from API
image_size = (28, 28)

def generate_class_images(name, max_drawings, recognized):
    directory = Path("data/quickdraw_images/" + name)
    if directory.exists():
        return
    
    directory.mkdir(parents=True)
    try:
        images = quickdraw.QuickDrawDataGroup(name, max_drawings=max_drawings, recognized=recognized, cache_dir="data/.quickdrawcache")
        for img in images.drawings:
            filename = directory.as_posix() + "/" + str(img.key_id) + ".png"
            img.get_image(stroke_width=3).resize(image_size).save(filename)
    except:
        return

def download_images_parallel(labels, max_drawings, recognized):
    # Use the number of CPUs or threads available
    max_threads = os.cpu_count() or 4  # Fallback to 4 if os.cpu_count() returns None
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = [
            executor.submit(generate_class_images, label, max_drawings, recognized)
            for label in labels
        ]
        for future in futures:
            future.result()

if __name__ == "__main__":
    categories = quickdraw.QuickDrawData().drawing_names
    #start = time.time()
    if (not modelExists):
        download_images_parallel(categories, max_drawings=1000, recognized=True)
    #end = time.time()
    #print(f"Image Download Time: {(end-start)//60}m {(end-start)%60}s")

# Picking Categories
random_categories = random.sample(categories, 12)
player_categories = {}
ai_categories = {}
ai_guesses = {}
for r in range(1, 4):
    ai_categories[r] = random_categories[r-1]
    player_categories[r] = (random_categories[3*r], random_categories[3*r+1], random_categories[3*r+2])
#print(player_categories, ai_categories)


if (not modelExists):
    # Splitting Dataset into Train/Validate Sets    
    dataset_dir = Path("data/quickdraw_images")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        color_mode="grayscale",
        image_size=image_size,
        batch_size=32
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        color_mode="grayscale",
        image_size=image_size,
        batch_size=32
    )

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(5000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Building CNN Model
    n_classes = 345
    input_shape = (28, 28, 1)

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

    # Converting the Model
    model.export('./models/CNN/base_model')
    converter = tf.lite.TFLiteConverter.from_saved_model("models/CNN/base_model")
    tflite_model = converter.convert()

    # Saving the Tflite Model.
    with open('models/CNN/tflite_model.tflite', 'wb') as f:
        f.write(tflite_model)

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="models/CNN/tflite_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function for CNN to guess player's drawing
def CNNguess(img_name):
    global ai_guesses
    # Preprocessing player's drawing
    player_img = Image.open(f"images/temp/player_temp/{img_name}.png")
    gray_player_img = player_img.convert('L')
    scaled_player_img = gray_player_img.resize(image_size)
    scaled_player_img.save(f"images/temp/test/{img_name}.png")

    input_data = tf.convert_to_tensor(np.array(scaled_player_img, dtype=np.float32))
    input_data = np.expand_dims(input_data, axis=(0, -1))
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Finding top 3 possible categories based on 3 highest probabilities
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    top3_categories = np.argpartition(output_data, -3)[-3:]
    top3_categories_sorted = top3_categories[np.argsort(output_data[top3_categories])[::-1]]
    for i in top3_categories_sorted:
        ai_guesses[categories[i]] = round(output_data[i]*100, 2)
    print(f"\nPredicted Categories: {ai_guesses}\n")
    
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

# Round Timer
roundTime = 60
counter, timerText = roundTime, str(roundTime).rjust(3)
TIMEREVENT = pygame.USEREVENT+3
pygame.time.set_timer(TIMEREVENT, 1000)
timerFont = pygame.font.SysFont('Courier New', 30)

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
            counter -= 1
            timerText = str(counter).rjust(3)
            # Decrementing timer
            if (counter >= 0):
                timerText = str(counter).rjust(3)
                if (playerTurn and (counter % 60 - 5) % 10 == 0):
                    imgName = f"{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())}_round_{roundNum}_{60-counter}s_player_drawing"
                    save(imgName, "temp/player_temp")
                    CNNguess(imgName)
                    ai_guesses = {}
            # Switching between player/ai turns when timer hits 0 and incrementing rounds
            elif ((counter < 0) and not(roundNum == 3 and not playerTurn)):
                playerTurn = not playerTurn
                if (playerTurn == True):
                    timerText = "Player's Turn!"
                    save(f"{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())}_round_{roundNum}_ai_drawing", "ai")
                    roundNum += 1
                else:
                    timerText = "AI's Turn!"
                    save(f"{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())}_round_{roundNum}_player_drawing", "player")
                drawColor = "#000000"
                screen.blit(timerFont.render(timerText, True, (0, 0, 0)), (32, 48))
                canvas.fill("#FFFFFF")
                pygame.display.flip()
                time.sleep(3)
                counter = roundTime + 3
            # Game finished after round 3
            # TODO: show points per round, total points for each team, and the winner
            else:
                gameOver = True
                timerText = "Game Over!"
                screen.blit(timerFont.render(timerText, True, (0, 0, 0)), (32, 48))
                pygame.display.flip()
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