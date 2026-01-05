"""
Generate the exercise extraction flowchart PNG.

Run this script to regenerate the flowchart after making changes:
    python generate_flowchart.py

The output flowchart.png is used by exercise_scanner.py to guide VLM extraction.
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def generate_flowchart():
    """Generate the exercise extraction flowchart."""
    # Create high quality flowchart (scaled up from 850x805)
    width, height = 1000, 950
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    # Fonts (larger sizes for crisp text)
    try:
        font_title = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22
        )
        font_section = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16
        )
        font_bold = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14
        )
        font_normal = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13
        )
        font_small = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12
        )
    except OSError:
        font_title = font_section = font_bold = font_normal = font_small = (
            ImageFont.load_default()
        )

    # Colors
    green_fill = "#90EE90"
    green_border = "#228B22"
    yellow_fill = "#FFFFE0"
    yellow_border = "#DAA520"
    cyan_fill = "#E0FFFF"
    cyan_border = "#008B8B"
    gray_fill = "#D3D3D3"
    gray_border = "#696969"
    dark_green = "#006400"
    red = "#CC0000"

    # Title (centered)
    draw.text(
        (width // 2 - 200, 15),
        "EXERCISE EXTRACTION FLOWCHART",
        fill="black",
        font=font_title,
    )

    # Step 1 box (green)
    x1, y1, x2, y2 = 300, 55, 700, 140
    draw.rounded_rectangle(
        [x1, y1, x2, y2], radius=20, fill=green_fill, outline=green_border, width=2
    )
    draw.text(
        (320, 65),
        "STEP 1: Find TOP-LEVEL exercises",
        fill="black",
        font=font_bold,
    )
    draw.text(
        (320, 88),
        'Marked: "Exercise 1", "Problem 1", "1)"',
        fill="black",
        font=font_normal,
    )
    draw.text(
        (320, 110),
        "Unmarked: distinct paragraphs",
        fill="black",
        font=font_normal,
    )

    # Arrow down from Step 1
    cx = 500  # center x for flowchart
    draw.line([(cx, 140), (cx, 170)], fill="black", width=2)
    draw.polygon([(cx - 5, 170), (cx + 5, 170), (cx, 182)], fill="black")

    # Step 2 box (gray)
    x1, y1, x2, y2 = 340, 185, 660, 235
    draw.rounded_rectangle(
        [x1, y1, x2, y2], radius=12, fill=gray_fill, outline=gray_border, width=2
    )
    draw.text((380, 200), "STEP 2: For each exercise", fill="black", font=font_bold)

    # STOP extraction text - right side
    draw.text((720, 175), "STOP extraction BEFORE:", fill=gray_border, font=font_bold)
    draw.text((720, 195), "• Form fields, answer blanks", fill="black", font=font_small)
    draw.text((720, 212), "• Solutions, answer sections", fill="black", font=font_small)
    draw.text((720, 229), "• Page headers/footers", fill="black", font=font_small)
    draw.text((720, 246), "• Junk text between exercises", fill="black", font=font_small)
    draw.text((720, 263), "• Exam-wide instructions", fill="black", font=font_small)
    draw.text((720, 280), "• General exam rules", fill="black", font=font_small)

    # Arrow down from Step 2
    draw.line([(cx, 235), (cx, 265)], fill="black", width=2)
    draw.polygon([(cx - 5, 265), (cx + 5, 265), (cx, 277)], fill="black")

    # Decision diamond (yellow)
    cy = 355
    dw, dh = 180, 80
    diamond = [(cx, cy - dh), (cx + dw, cy), (cx, cy + dh), (cx - dw, cy)]
    draw.polygon(diamond, fill=yellow_fill, outline=yellow_border, width=2)
    draw.text((cx - 50, cy - 45), "Does it contain", fill="black", font=font_small)
    draw.text((cx - 105, cy - 22), "SEPARATE TASKS requiring", fill="black", font=font_bold)
    draw.text((cx - 90, cy + 5), "SEPARATE ANSWERS?", fill="black", font=font_bold)

    # YES label with background box and arrow (left)
    yes_x, yes_y = 285, 310
    draw.rounded_rectangle(
        [yes_x, yes_y, yes_x + 45, yes_y + 24], radius=4, fill=green_fill, outline=green_border, width=1
    )
    draw.text((yes_x + 8, yes_y + 4), "YES", fill=green_border, font=font_bold)
    draw.line([(cx - dw, cy), (220, cy)], fill="black", width=2)
    draw.line([(220, cy), (220, 460)], fill="black", width=2)
    draw.polygon([(215, 460), (225, 460), (220, 472)], fill="black")

    # NO label with background box and arrow (right)
    no_x, no_y = 685, 310
    draw.rounded_rectangle(
        [no_x, no_y, no_x + 38, no_y + 24], radius=4, fill="#FFCCCC", outline=red, width=1
    )
    draw.text((no_x + 8, no_y + 4), "NO", fill=red, font=font_bold)
    draw.line([(cx + dw, cy), (780, cy)], fill="black", width=2)
    draw.line([(780, cy), (780, 460)], fill="black", width=2)
    draw.polygon([(775, 460), (785, 460), (780, 472)], fill="black")

    # PARENT box (cyan) - left
    x1, y1, x2, y2 = 100, 475, 340, 550
    draw.rounded_rectangle(
        [x1, y1, x2, y2], radius=12, fill=cyan_fill, outline=cyan_border, width=2
    )
    draw.text((175, 485), "Extract:", fill="black", font=font_normal)
    draw.text((135, 505), "PARENT (full setup)", fill="black", font=font_bold)
    draw.text((145, 527), "+ SUB-QUESTIONS", fill="black", font=font_bold)

    # PARENT TEXT - below cyan box
    draw.text((100, 560), "PARENT TEXT = FULL exercise block:", fill=cyan_border, font=font_bold)
    draw.text((100, 580), "• Intro + sub-questions + any text after", fill="black", font=font_small)
    draw.text((100, 597), "• Scenario/problem setup", fill="black", font=font_small)
    draw.text((100, 614), "• Definitions, formulas, given values", fill="black", font=font_small)
    draw.text((100, 631), "• Data that sub-questions reference", fill="black", font=font_small)

    # STANDALONE box (green) - right
    x1, y1, x2, y2 = 680, 475, 880, 545
    draw.rounded_rectangle(
        [x1, y1, x2, y2], radius=12, fill=green_fill, outline=green_border, width=2
    )
    draw.text((720, 492), "STANDALONE", fill="black", font=font_bold)
    draw.text((745, 515), "exercise", fill="black", font=font_bold)

    # Separator line
    draw.line([(30, 665), (970, 665)], fill="#AAAAAA", width=1)

    # KEY DISTINCTIONS - styled table
    table_y = 680
    table_left = 30
    table_right = 970
    table_mid = 500  # Centered between table_left and table_right
    row_height = 24
    title_height = 26
    info_height = 85
    header_height = 28

    # Title row - KEY DISTINCTIONS
    draw.rectangle(
        [table_left, table_y, table_right, table_y + title_height],
        fill="#4A4A4A",
        outline="#333333",
        width=1,
    )
    draw.text(
        (width // 2 - 70, table_y + 5),
        "KEY DISTINCTIONS",
        fill="white",
        font=font_bold,
    )

    # "Exercise parts can look like:" info section
    info_y = table_y + title_height
    draw.rectangle(
        [table_left, info_y, table_right, info_y + info_height],
        fill="#FFFEF0",
        outline="#888888",
        width=1,
    )
    draw.text(
        (width // 2 - 100, info_y + 5),
        "Exercise parts can look like:",
        fill="#555555",
        font=font_bold,
    )
    # Vertical centered list of bullets
    bullet_x = width // 2 - 150  # Centered (accounting for text width)
    draw.text(
        (bullet_x, info_y + 22),
        "• Marked: a), b), c), 1., 2., i), ii), -",
        fill="black",
        font=font_small,
    )
    draw.text(
        (bullet_x, info_y + 36),
        "• Unmarked: separate tasks asking different things",
        fill="black",
        font=font_small,
    )
    draw.text(
        (bullet_x, info_y + 50),
        '• Inline: "(a)...(b)..." or "Explain X... Calculate Y..."',
        fill="black",
        font=font_small,
    )
    draw.text(
        (bullet_x, info_y + 64),
        "• Nested: 1a, 1b or 1.1, 1.2... or on separate lines",
        fill="black",
        font=font_small,
    )

    # Column headers
    col_header_y = info_y + info_height
    draw.rectangle(
        [table_left, col_header_y, table_right, col_header_y + header_height],
        fill="#E8E8E8",
        outline="#888888",
        width=1,
    )
    draw.text(
        (table_left + 100, col_header_y + 6),
        "SUB-QUESTION (split these)",
        fill=dark_green,
        font=font_bold,
    )
    draw.text(
        (table_mid + 60, col_header_y + 6),
        "NOT a sub-question (keep together)",
        fill=red,
        font=font_bold,
    )
    # Header divider
    draw.line(
        [(table_mid, col_header_y), (table_mid, col_header_y + header_height)],
        fill="#888888",
        width=1,
    )

    # Update table_y for rows to start after info section and header
    table_y = col_header_y

    # Table rows (reduced - decision criteria only)
    rows_left = [
        "SEPARATE TASKS requiring SEPARATE ANSWERS",
        "Each tests a DIFFERENT skill",
        "Asks to DO: Find, Calculate, Prove, Explain...",
        "",
    ]
    rows_right = [
        "GIVES information (setup, definitions, given data)",
        "Same skill, different inputs/cases → ONE answer",
        "Multiple choice options (A/B/C/D) - ONE exercise",
        "All parts contribute to ONE answer",
    ]

    for i, (left, right) in enumerate(zip(rows_left, rows_right)):
        row_y = table_y + header_height + i * row_height
        # Alternating row background
        bg_color = "#F8F8F8" if i % 2 == 0 else "white"
        draw.rectangle(
            [table_left, row_y, table_right, row_y + row_height],
            fill=bg_color,
            outline="#CCCCCC",
            width=1,
        )
        # Column divider
        draw.line(
            [(table_mid, row_y), (table_mid, row_y + row_height)],
            fill="#CCCCCC",
            width=1,
        )
        # Row text
        draw.text((table_left + 8, row_y + 5), left, fill="black", font=font_small)
        if right:
            draw.text((table_mid + 8, row_y + 5), right, fill="black", font=font_small)

    # Save to same directory as this script
    output_path = Path(__file__).parent / "flowchart.png"
    img.save(output_path, quality=95)
    print(f"Flowchart saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    generate_flowchart()
