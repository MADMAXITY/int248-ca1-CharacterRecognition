import pygame, sys
from pygame.locals import *
import cv2
import numpy as np
import tensorflow
import keras
import tkinter as tk
from tkinter import *
import tkinter.font as font
from tkinter import messagebox


mapped_labels = {
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "A",
    11: "B",
    12: "C",
    13: "D",
    14: "E",
    15: "F",
    16: "G",
    17: "H",
    18: "I",
    19: "J",
    20: "K",
    21: "L",
    22: "M",
    23: "N",
    24: "O",
    25: "P",
    26: "Q",
    27: "R",
    28: "S",
    29: "T",
    30: "U",
    31: "V",
    32: "W",
    33: "X",
    34: "Y",
    35: "Z",
    36: "a",
    37: "b",
    38: "d",
    39: "e",
    40: "f",
    41: "g",
    42: "h",
    43: "n",
    44: "q",
    45: "r",
    46: "t",
}


def main():
    pygame.init()

    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)

    mouse_position = (0, 0)
    drawing = False
    screen = pygame.display.set_mode((500, 500), 0, 32)
    screen.fill(WHITE)

    image_pixels = []
    pygame.display.set_caption("Drawing Board")
    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                # pygame.quit()
                # sys.exit()
                messagebox.showwarning(
                    "warning",
                    "Please press q to exit, quitting like this can cause some issues.",
                )
            if event.type == pygame.KEYDOWN:
                if event.key == K_SPACE:
                    screen.fill(WHITE)
                if event.key == K_q:
                    pygame.quit()
                if event.key == K_p:

                    for i in range(500):
                        temp = []
                        for j in range(500):
                            if screen.get_at((j, i)) == (0, 0, 0, 255):
                                temp.append(255)
                            else:
                                temp.append(0)
                        image_pixels.append(temp)
                    pygame.quit()
                    return image_pixels
            elif event.type == MOUSEMOTION:
                if drawing:
                    mouse_position = pygame.mouse.get_pos()
                    pygame.draw.circle(screen, BLACK, mouse_position, 10)
                    x, y = mouse_position
            elif event.type == MOUSEBUTTONUP:
                mouse_position = (0, 0)
                drawing = False
            elif event.type == MOUSEBUTTONDOWN:
                drawing = True
        try:
            pygame.display.update()
        except:
            break


def operations():
    model = keras.models.load_model("SavedModels/handwritten-cnn-trained.h5")
    image_pixels = main()

    img = np.uint8(image_pixels)

    # cv2.imshow("Captured Image", img)

    resized = cv2.resize(img, (28, 28), interpolation=cv2.INTER_LINEAR)
    reshaped = resized.copy()
    reshaped = np.uint8(reshaped).reshape(-1, 28, 28, 1)
    prediction = model.predict(reshaped)
    val = max(prediction[0])
    value = ""
    for i in range(len(prediction[0])):
        if prediction[0][i] == val:
            value = i

    editz = cv2.putText(
        img,
        "Prediction : {}".format(mapped_labels[value]),
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.imshow("Editz", editz)
    # cv2.imshow("resized", resized)

    cv2.waitKey()
    cv2.destroyAllWindows()


def exitor():
    sys.exit()


if __name__ == "__main__":

    window = tk.Tk()
    window.geometry("500x500")
    window.title("Character recognition")
    window.configure(bg="gray25")

    fonting = font.Font(family="Times New Roman", size=20)
    lab1 = Label(
        window,
        text="Character recognition",
        bg="gray25",
        fg="ghost white",
        font=("Helventica", 14),
    ).place(x=50, y=50)

    Button(
        window,
        text="Open Drawing Window",
        command=operations,
        fg="ghost white",
        bg="gray20",
    ).place(x=50, y=120)

    Label(
        text="Note : After drawing press p to see the prediction.",
        bg="gray25",
        fg="ghost white",
        font=("Helventica", 12),
    ).place(x=50, y=170)
    Label(
        text="Note : Press Space key to clear the drawing window.",
        bg="gray25",
        fg="ghost white",
        font=("Helventica", 12),
    ).place(x=50, y=200)

    Label(
        text="If the drawing window or prediction window is open,\n the press q to exit.Don't use the cross!",
        bg="gray25",
        fg="ghost white",
        font=("Helventica", 12),
    ).place(x=50, y=250)

    Button(
        window,
        text="Exit",
        command=exitor,
        fg="ghost white",
        bg="gray20",
    ).place(x=230, y=450)
    window.mainloop()
