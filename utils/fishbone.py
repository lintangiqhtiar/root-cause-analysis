import math
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Wedge
import json
from utils.glda import hasil_lda
import streamlit as st
# Fixed fishbone diagram generator
def generate_diagram(file_path):
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        st.error(f"File {file_path} tidak ditemukan.")
        return None
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.axis('off')
    
    # Define the problems function
    def problems(data, problem_x, problem_y, angle_x, angle_y):
        """
        Draw each problem section of the Ishikawa plot.
        """
        ax.annotate(str.upper(data), xy=(problem_x, problem_y),
                    xytext=(angle_x, angle_y),
                    fontsize=10,
                    color='white',
                    weight='bold',
                    xycoords='data',
                    verticalalignment='center',
                    horizontalalignment='center',
                    textcoords='offset fontsize',
                    arrowprops=dict(arrowstyle="->", facecolor='black'),
                    bbox=dict(boxstyle='square',
                              facecolor='tab:blue',
                              pad=0.8))
    
    # Define the causes function
    def causes(data, cause_x, cause_y, cause_xytext=(-9, -0.3), top=True):
        """
        Place each cause to a position relative to the problems annotations.
        """
        for index, cause in enumerate(data):
            # [<x pos>, <y pos>]
            coords = [[0.02, 0],
                      [0.23, 0.5],
                      [-0.46, -1],
                      [0.69, 1.5],
                      [-0.92, -2],
                      [1.15, 2.5]]
            
            # First 'cause' annotation is placed in the middle of the 'problems' arrow
            # and each subsequent cause is plotted above or below it in succession.
            cause_x_pos = cause_x - coords[index][0]
            cause_y_pos = cause_y + (coords[index][1] if top else -coords[index][1])
            
            ax.annotate(cause, xy=(cause_x_pos, cause_y_pos),
                        horizontalalignment='center',
                        xytext=cause_xytext,
                        fontsize=9,
                        xycoords='data',
                        textcoords='offset fontsize',
                        arrowprops=dict(arrowstyle="->",
                                        facecolor='black'))
    
    # Define the draw_spine function
    def draw_spine(xmin, xmax):
        """
        Draw main spine, head and tail.
        """
        # draw main spine
        ax.plot([xmin - 0.1, xmax], [0, 0], color='tab:blue', linewidth=2)
        # draw fish head
        ax.text(xmax + 0.1, - 0.05, 'PROBLEM', fontsize=10,
                weight='bold', color='white')
        semicircle = Wedge((xmax, 0), 1, 270, 90, fc='tab:blue')
        ax.add_patch(semicircle)
        # draw fish tail
        tail_pos = [[xmin - 0.8, 0.8], [xmin - 0.8, -0.8], [xmin, -0.01]]
        triangle = Polygon(tail_pos, fc='tab:blue')
        ax.add_patch(triangle)
    
    # Define the draw_body function
    def draw_body(data):
        """
        Place each problem section in its correct place by changing
        the coordinates on each loop.
        """
        # Set the length of the spine according to the number of 'problem' categories.
        length = (len(data) // 2)
        draw_spine(-2 - length, 2 + length)
        
        # Change the coordinates of the 'problem' annotations after each one is rendered.
        offset = 0
        prob_section = [1.55, 0.8]
        for index, (problem_key, problem_values) in enumerate(data.items()):
            plot_above = index % 2 == 0
            cause_arrow_y = 1.7 if plot_above else -1.7
            y_prob_angle = 16 if plot_above else -16
            
            # Plot each section in pairs along the main spine.
            prob_arrow_x = prob_section[0] + length + offset
            cause_arrow_x = prob_section[1] + length + offset
            if not plot_above:
                offset -= 2.5
            if index > 5:
                raise ValueError(f'Maximum number of problems is 6, you have entered {len(data)}')
            
            problems(problem_key, prob_arrow_x, 0, -12, y_prob_angle)
            causes(problem_values, cause_arrow_x, cause_arrow_y, top=plot_above)
    
    # Draw the fishbone diagram
    draw_body(data)
    
    # Return the figure instead of showing it
    return fig