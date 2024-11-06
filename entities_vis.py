import spacy

file = "/user/work/ep16475/llama/my-llama-models/MRI Lumbosacral Spine/OU43548/F report.txt"

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

def color_code(text1, text2, arr1, text3,text4,arr2, outfile):
    # Split the texts into words
    text1_split = text1.split(" ")
    text2_split = text2.split(" ")
    text3_split = text3.split(" ")
    text4_split = text4.split(" ")

    # Create figure and four subplots arranged in a 2x2 grid
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 20))  # Increased height for four subplots
    ax1.axis("off")
    ax2.axis("off")
    ax3.axis("off")
    ax4.axis("off")

    # Starting y position for both texts
    start_y = 1  # Start closer to the top
    line_spacing = 0.05  # Decreased line spacing for more lines on each subplot

    # Color coding for words
    bbox1 = dict(boxstyle="round,pad=0.3", fc="white", ec="g", lw=2)
    bbox2 = dict(boxstyle="round,pad=0.3", fc="white", ec="y", lw=2)
    bbox3 = dict(boxstyle="round,pad=0.3", fc="white", ec="r", lw=2)
    bbox4 = dict(boxstyle="round,pad=0.3", fc="white", ec="b", lw=2)

    def plot_text(ax, text_split, report_type, array):
        x = 0  # Initial x position for each word
        y = start_y  # Initial y position for the first line
        l1 = array[0]
        l2 = array[1]
        l3 = array[2]
        l4 = array[3]
        for word in text_split:
            # Determine color and highlight for each word
            if word in l1 or word.replace('.', '') in l1 or word.lower() in l1:
                ax.text(x, y, word, color="black", bbox=bbox1, transform=ax.transAxes)
            elif word in l2 or word.replace('.', '') in l2 or word.lower() in l2:
                ax.text(x, y, word, color="black", bbox=bbox2, transform=ax.transAxes)
            elif (word in l3 or word.replace('.', '') in l3 or word.lower() in l3) and report_type == "final":
                ax.text(x, y, word, color="black", bbox=bbox3, transform=ax.transAxes)
            elif (word in l4 or word.replace('.', '') in l4 or word.lower() in l4) and report_type == "prelim":
                ax.text(x, y, word, color="black", bbox=bbox4, transform=ax.transAxes)
            else:
                ax.text(x, y, word, transform=ax.transAxes)

            # Move x position for next word and wrap if needed
            x += len(word) * 0.02   # Adjusted whitespace
            if x >= 0.95:  # Wrap text to next line if x position exceeds 0.9 in axis scale
                x = 0
                y -= line_spacing  # Move down for new line

                # If y position reaches the bottom, start a new column within the subplot
                if y < 0.05:
                    y = start_y
                    x += 0.45  # Start a new column further to the right within the same subplot

    # Plot text on each subplot
    plot_text(ax1, text1_split, "final", arr1)
    plot_text(ax2, text2_split, "prelim", arr1)
    plot_text(ax3, text3_split, "final", arr2)
    plot_text(ax4, text4_split, "prelim", arr2)

    # Save and show figure
    plt.savefig(outfile)
    plt.show()

def color_words(text1, array1, text2, array2, text3, array3, text4, array4, outfile):
    # Split the texts into words
    text1_split = text1.split(" ")
    text2_split = text2.split(" ")
    text3_split = text3.split(" ")
    text4_split = text4.split(" ")

    # Create figure and four subplots arranged in a 2x2 grid
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 20))  # Increased height for four subplots
    ax1.axis("off")
    ax2.axis("off")
    ax3.axis("off")
    ax4.axis("off")

    # Starting y position for all texts
    start_y = 1  # Start closer to the top
    line_spacing = 0.05  # Line spacing

    # Color coding for words
    bbox1 = dict(boxstyle="round,pad=0.3", fc="white", ec="g", lw=2)
    bbox2 = dict(boxstyle="round,pad=0.3", fc="white", ec="y", lw=2)
    bbox3 = dict(boxstyle="round,pad=0.3", fc="white", ec="r", lw=2)
    bbox4 = dict(boxstyle="round,pad=0.3", fc="white", ec="b", lw=2)

    def plot_text(ax, text_split, array):
        x = 0  # Initial x position for each word
        y = start_y  # Initial y position for the first line

        for i, word in enumerate(text_split):
            # Determine color and highlight for each word
            if array[i] == 1:
                ax.text(x, y, word, color="black", bbox=bbox1, transform=ax.transAxes)
            elif array[i] == 2:
                 ax.text(x, y, word, color="black", bbox=bbox2, transform=ax.transAxes)
            elif array[i] == 3:
                ax.text(x, y, word, color="black", bbox=bbox3, transform=ax.transAxes)
            elif array[i] == 4:
                ax.text(x, y, word, color="black", bbox=bbox4, transform=ax.transAxes)
            else:
                ax.text(x, y, word, transform=ax.transAxes)

            # Move x position for next word and wrap if needed
            x += len(word) * 0.02   # Adjusted whitespace
            if x >= 0.95:  # Wrap text to next line if x position exceeds 0.9 in axis scale
                x = 0
                y -= line_spacing  # Move down for new line

                # If y position reaches the bottom, start a new column within the subplot
                if y < 0.05:
                    y = start_y
                    x += 0.45  # Start a new column further to the right within the same subplot

    # Plot text on each subplot
    plot_text(ax1, text1_split, array1)
    plot_text(ax2, text2_split, array2)
    plot_text(ax3, text3_split, array3)
    plot_text(ax4, text4_split, array4)

    # Save and show figure
    plt.savefig(outfile)
    plt.show()



